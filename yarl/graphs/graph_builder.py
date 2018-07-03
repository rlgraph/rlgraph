# Copyright 2018 The YARL-Project, All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import logging

from yarl import YARLError, Specifiable, get_backend
from yarl.components import Component
from yarl.spaces import Space
from yarl.spaces.space_utils import get_space_from_op
from yarl.utils.input_parsing import parse_summary_spec
from yarl.utils.util import force_list, force_tuple
from yarl.utils.ops import FlattenedDataOp, DataOpRecord, DataOpRecordColumnIntoGraphFn, DataOpRecordColumnFromGraphFn
from yarl.utils.component_printout import component_print_out

if get_backend() == "tf":
    import tensorflow as tf


class GraphBuilder(Specifiable):
    """
    The graph builder assembles the YARL meta-graph by tracing through
    components, sockets and connections and creating the underlying computation
    graph.
    """
    # Break iterative DFS graph building if caught in a loop.
    MAX_ITERATIVE_LOOPS = 100

    def __init__(self, name="model", action_space=None, summary_spec=None):
        """
        Args:
            name (str): The name of this GraphBuilder and of the meta-graph's core component.
            action_space (Optional[Space]): The action Space information to be passed into calls to each Components'
                `when_input_complete` methods.
            summary_spec (Optional[dict]): A specification dict that defines, which summaries we would like to
                create in the graph and register with each Component.
        """
        # The name of this model. Our core Component gets this name.
        self.logger = logging.getLogger(__name__)
        self.name = name
        self.action_space = action_space
        self.summary_spec = parse_summary_spec(summary_spec)

        # All components assigned to each device, for debugging and analysis.
        self.device_component_assignments = dict()
        self.available_devices = None
        self.default_device = None
        self.device_strategy = None

        # Counting recursive steps.
        self.build_steps = 0

        # Create an empty core Component into which everything will be assembled by an Algo.
        self.core_component = None  # Component(name=self.name, is_core=True)

        # Maps API method names to in- (placeholders) and out op columns (ops to pull).
        self.api = dict()

        # Dict of unprocessed (and complete) op-record columns by key=op-column ID.
        # Columns that have been forwarded will be erased again from this collection.
        self.unprocessed_in_op_columns = OrderedDict()
        self.unprocessed_complete_in_op_columns = OrderedDict()

    def build(self, input_spaces, available_devices, default_device=None, device_strategy='default'):
        """
        Builds the actual backend-specific graph from the YARL metagraph.
        Loops through all our sub-components starting at core and assembles the graph by creating placeholders,
        following Socket->Socket connections and running through our GraphFunctions to generate DataOps.

        Args:
            input_spaces (dict):
            available_devices (list): Devices which can be used to assign parts of the graph
                during graph assembly.
            default_device (Optional[str]): Default device identifier.
            device_strategy (Optional[str]): Device strategy.
        """
        # Build the meta-graph (generating empty op-record columns around API methods
        # and graph_fns).
        self.build_meta_graph(input_spaces)

        # self.sanity_check_meta_graph()

        # Set devices usable for this graph.
        self.available_devices = available_devices
        self.device_strategy = device_strategy
        self.default_device = default_device

        # Push all spaces through the API methods, then enter the main iterative DFS while loop.
        op_records_to_process = self.build_input_space_ops(input_spaces)

        self.build_graph(op_records_to_process)

    def build_meta_graph(self, input_spaces=None):
        """
        Builds the meta graph by building op records for all api methods.
        Args:
            input_spaces (Optional[Space]): Input spaces for api methods.
        """
        # Sanity check input_spaces dict.
        if input_spaces is not None:
            for api_method_name in input_spaces.keys():
                if api_method_name not in self.core_component.api_methods:
                    raise YARLError("ERROR: `input_spaces` contains API-method ('{}') that's not defined in "
                                    "core-component '{}'!".format(api_method_name, self.core_component.name))

        # Call all API methods of the core and thereby, create empty in-op columns that serve as placeholders
        # and directed links for the build time.
        self.logger.debug(self.core_component.api_methods)
        for api_method_name, api_method_rec in self.core_component.api_methods.items():
            # Create an new in column and map it to the resulting out column.
            in_ops_records = list()
            if input_spaces is not None and api_method_name in input_spaces:
                for i in range(len(force_list(input_spaces[api_method_name]))):
                    in_ops_records.append(DataOpRecord(description="input-{}-{}".format(api_method_name, i)))
            self.core_component.call(api_method_rec.method, *in_ops_records)
            # Register interface.
            self.api[api_method_name] = (in_ops_records, api_method_rec.out_op_columns[0].op_records)

    def sanity_check_meta_graph(self, component=None):
        """
        TODO: Rewrite this method according to new API-method design.

        Checks whether all the `component`'s and its sub-components' in-Sockets are simply connected in the
        meta-graph and raises detailed error messages if not. A connection to an in-Socket is ok if ...
        a) it's coming from another Socket or
        b) it's coming from a Space object

        Args:
            component (Component): The Component to analyze for incoming connections.
        """
        component = component or self.core_component

        if self.logger.level <= logging.INFO:
            component_print_out(component)

        # Check all the Component's in-Sockets for being connected from a Space/Socket.
        for in_sock in component.input_sockets:  # type: Socket
            if len(in_sock.incoming_connections) == 0 and \
                    in_sock.name not in component.unconnected_sockets_in_meta_graph:
                raise YARLError("Component '{}' has in-Socket ({}) without any incoming connections! If this is "
                                "intended before the build process, you have to add the Socket's name to the "
                                "Component's `unconnected_sockets_in_meta_graph` set. Then this error will be "
                                "suppressed for this Component.".format(component.name, in_sock.name))

        # Check all the component's graph_fns for input-completeness.
        for graph_fn in component.graph_fns:  # type: GraphFunction
            for in_sock_rec in graph_fn.input_sockets.values():
                in_sock = in_sock_rec["socket"]
                if len(in_sock.incoming_connections) == 0 and \
                        in_sock.name not in component.unconnected_sockets_in_meta_graph:
                    raise YARLError("GraphFn {}/{} has in-Socket ({}) without any incoming "
                                    "connections!".format(component.name, graph_fn.name, in_sock_rec["socket"].name))

        # Recursively call this method on all the sub-component's sub-components.
        for sub_component in component.sub_components.values():
            self.build_steps += 1
            if self.build_steps >= self.MAX_ITERATIVE_LOOPS:
                raise YARLError("Error sanity checking graph, reached max recursion steps: {}".format(
                    self.MAX_ITERATIVE_LOOPS
                ))
            self.sanity_check_meta_graph(sub_component)

    def build_input_space_ops(self, input_spaces):
        """
        Generates ops from Space information and stores these ops in the DataOpRecords of our API
        methods.

        Args:
            input_spaces (dict): Dict with keys=api-method names; values=list of Space objects or specification dicts
                to create the Spaces that go into the APIMethodRecords.

        Returns:
            Set[DataOpRecord]: A set of DataOpRecords with which we should start the building
                process.
        """
        op_records_to_process = set()

        for api_method_name, (in_op_records, _) in self.api.items():
            spaces = force_list(input_spaces[api_method_name]) if input_spaces is not None and \
                                                                  api_method_name in input_spaces else list()
            assert len(spaces) == len(in_op_records)

            # Create the placeholder and store it in the given DataOpRecords.
            for i, space in enumerate(spaces):
                if not isinstance(space, Space):
                    space = Space.from_spec(space)
                # Generate a placeholder and store it as well as the Space itself.
                in_op_records[i].op = space.get_tensor_variable(name="api-"+api_method_name+"/param-"+str(i),
                                                                is_input_feed=True)
                in_op_records[i].space = space

                op_records_to_process.add(in_op_records[i])

        return op_records_to_process

    def build_graph(self, op_records_to_process):
        """
        The actual iterative depth-first search algorithm to build our graph from the already existing
        meta-Graph structure.
        Starts from a set of DataOpRecords populated with the initial placeholders (from input
        Spaces). Keeps pushing these ops through the meta-graph until a non-complete graph_fn
        or a non-complete Component (with at least one non-complete API-method) is reached.
        Replaces the ops in the set with the newly reached ones and re-iterates like this until all op-records
        in the entire meta-graph have been filled with actual ops.

        Args:
            op_records_to_process (Set[DataOpRecord]): The initial set of DataOpRecords (with populated `op` fields
                to start the build with).
        """
        # Tag very last out-op-records with op="last", so we know in the build process that we are done.
        for _, out_op_records in self.api.values():
            for out_op_rec in out_op_records:
                out_op_rec.op = "done"

        # Collect all components and do those first that are already input-complete.
        components = self.get_all_components()
        for component in components:
            self.build_component_when_input_complete(component, op_records_to_process)

        # Re-iterate until our bag of op-recs to process is empty.
        loop_counter = 0
        while len(op_records_to_process) > 0:
            op_records_list = list(op_records_to_process)
            new_op_records_to_process = set()
            for op_rec in op_records_list:  # type: DataOpRecord
                # There are next records:
                if len(op_rec.next) > 0:
                    # Push the op-record forward one step.
                    for next_op_rec in op_rec.next:  # type: DataOpRecord
                        # If not last op in this API-method ("done") -> continue.
                        # Otherwise, replace "done" with actual op.
                        if next_op_rec.op != "done":
                            assert next_op_rec.op is None
                            new_op_records_to_process.add(next_op_rec)
                        # Push op and Space into next op-record.
                        next_op_rec.op = op_rec.op
                        next_op_rec.space = op_rec.space

                        # Did we enter a new Component? If yes, check input-completeness and
                        # - If op_rec.column is None -> We are at the very beginning of the graph (op_rec.op is a
                        # placeholder).
                        if op_rec.column is None or op_rec.column.component is not next_op_rec.column.component:
                            self.build_component_when_input_complete(next_op_rec.column.component, new_op_records_to_process)

                # No next records: Ignore columns coming from a _variables graph_fn and not going anywhere.
                elif not isinstance(op_rec.column, DataOpRecordColumnFromGraphFn) or \
                        op_rec.column.graph_fn_name != "_graph_fn__variables":
                    # Must be a into graph_fn column.
                    assert isinstance(op_rec.column, DataOpRecordColumnIntoGraphFn)
                    # If column complete, call the graph_fn.
                    if op_rec.column.is_complete():
                        # Call the graph_fn with the given column and call-options.
                        self.run_through_graph_fn_with_device_and_scope(op_rec.column)
                        # Store all resulting op_recs (returned by the graph_fn) to be processed next.
                        new_op_records_to_process.update(op_rec.column.out_graph_fn_column.op_records)
                    # If column incomplete, stop here for now.
                    else:
                        new_op_records_to_process.add(op_rec)

            # Replace all old records with new ones.
            for op_rec in op_records_list:
                op_records_to_process.remove(op_rec)
            for op_rec in new_op_records_to_process:
                op_records_to_process.add(op_rec)

            # Sanity check iterative loops.
            loop_counter += 1
            if loop_counter >= self.MAX_ITERATIVE_LOOPS:
                raise YARLError("Sanity checking graph-build procedure: Reached max iteration steps: {}!".format(
                    self.MAX_ITERATIVE_LOOPS
                ))

    def get_all_components(self, component=None, list_=None, level_=0):
        """
        Returns all Components of the core-Component sorted by their nesting-level (... grand-children before children
        before parents).

        Args:
            component (Optional[Component]): The Component to look through. None for the core-Component.
            list_ (Optional[List[Component]])): A list of already collected components to append to.

        Returns:
            List[Component]: A list with all the components in `component`.
        """
        component = component or self.core_component
        return_ = False
        if list_ is None:
            list_ = dict()
            return_ = True
        if level_ not in list_:
            list_[level_] = list()
        list_[level_].append(component)
        level_ += 1
        for sub_component in component.sub_components.values():
            self.get_all_components(sub_component, list_, level_)
        if return_:
            ret = list()
            for l in sorted(list_.keys(), reverse=True):
                ret.extend(list_[l])
            return ret

    def build_component_when_input_complete(self, component, op_records_to_process):
        # Not input complete yet -> Check now.
        if component.input_complete is False:
            spaces_dict = component.check_input_completeness()
            # call `when_input_complete` once on that Component.
            if spaces_dict is not None:
                self.logger.debug("Component {} is input-complete; spaces_dict={}".
                                  format(component.name, spaces_dict))
                no_input_graph_fn_columns = component.when_input_complete(spaces_dict,
                                                                          self.action_space)
                # Call all no-input graph_fns of the new Component.
                for no_in_col in no_input_graph_fn_columns:
                    self.run_through_graph_fn_with_device_and_scope(no_in_col)
                    # Keep working with the generated output ops.
                    op_records_to_process.update(no_in_col.out_graph_fn_column.op_records)

        # Check variable-completeness and run the _variable graph_fn if the component just became "variable-complete".
        if component.input_complete is True and component.variable_complete is False and \
                component.check_variable_completeness():
            component.call(component._graph_fn__variables)
            graph_fn_rec = component.graph_fns["_graph_fn__variables"]
            self.run_through_graph_fn_with_device_and_scope(graph_fn_rec.in_op_columns[0])
            # Keep working with the generated output ops.
            op_records_to_process.update(graph_fn_rec.out_op_columns[0].op_records)

    def run_through_graph_fn_with_device_and_scope(self, op_rec_column):
        """
        Runs through a graph_fn with the given ops and thereby assigns a device (Component's device or GraphBuilder's
        default) to the ops generated by a graph_fn.

        Args:
            op_rec_column (DataOpRecordColumnIntoGraphFn): The column of DataOpRecords to be fed through the
                graph_fn.
        """
        # We have to specify the device and the variable scope here as we will be running through a
        # GraphFunction, which may add ops to the graph.
        assigned_device = op_rec_column.component.device or self.default_device

        if get_backend() == "tf":
            if assigned_device is not None and assigned_device not in self.available_devices:
                self.logger.error("Assigned device '{}' for graph_fn '{}' not in available devices:\n {}".
                                  format(assigned_device, op_rec_column.graph_fn.__name__, self.available_devices))

            if assigned_device is not None:
                # These strategies always set a default device.
                assert self.device_strategy == 'default' or self.device_strategy == 'multi_gpu_sync'
                # Assign proper device to all ops created in this context manager.
                with tf.device(assigned_device):
                    # Name ops correctly according to our Component hierarchy.
                    with tf.name_scope(op_rec_column.component.global_scope+
                                       ('/' if op_rec_column.component.global_scope else "")):
                        self.logger.debug(
                            "Assigning device '{}' to graph_fn '{}' (scope '{}').".
                                format(assigned_device, op_rec_column.graph_fn.__name__, op_rec_column.component.global_scope)
                        )
                        self.run_through_graph_fn(op_rec_column)
            else:
                # Custom device strategy with no default device.
                assert self.device_strategy == 'custom'
                # Name ops correctly according to our Component hierarchy.
                with tf.name_scope(op_rec_column.component.global_scope +
                                   ('/' if op_rec_column.component.global_scope else "")):
                    self.logger.debug(
                        "Assigning device '{}' to graph_fn '{}' (scope '{}').".
                            format(assigned_device, op_rec_column.graph_fn.__name__,
                                   op_rec_column.component.global_scope)
                    )
                    self.run_through_graph_fn(op_rec_column)

        # Store assigned names for debugging.
        if assigned_device is not None:
            if assigned_device not in self.device_component_assignments:
                self.device_component_assignments[assigned_device] = [str(op_rec_column.graph_fn.__name__)]
            else:
                self.device_component_assignments[assigned_device].append(str(op_rec_column.graph_fn.__name__))

    @staticmethod
    def run_through_graph_fn(op_rec_column):
        """
        Pushes all ops in the column through the respective graph_fn (graph_fn-spec and call-options are part of
        the column).
        Call options include flattening ops, flattening+splitting ops and (when splitting) adding the auto-generated
        flat key as first parameter to the different (split) calls of graph_fn.
        After the call

        Args:
            op_rec_column (DataOpRecordColumnIntoGraphFn): The column of DataOpRecords to be fed through the
                graph_fn.
        """
        in_ops = [r.op for r in op_rec_column.op_records]
        assert all(op is not None for op in in_ops)  # just make sure

        # Build the ops from this input-combination.
        # Flatten input items.
        if op_rec_column.flatten_ops is not False:
            flattened_ops = op_rec_column.flatten_input_ops(*in_ops)
            # Split into SingleDataOps?
            if op_rec_column.split_ops:
                call_params = op_rec_column.split_flattened_input_ops(*flattened_ops)
                # There is some splitting to do. Call graph_fn many times (one for each split).
                if isinstance(call_params, FlattenedDataOp):
                    ops = dict()
                    num_return_values = -1
                    for key, params in call_params.items():
                        ops[key] = force_tuple(op_rec_column.graph_fn(*params))
                        if num_return_values >= 0 and num_return_values != len(ops[key]):
                            raise YARLError("Different split-runs through {} do not return the same number of "
                                            "values!".format(op_rec_column.graph_fn.__name__))
                        num_return_values = len(ops[key])
                    # Un-split the results dict into a tuple of `num_return_values` slots.
                    un_split_ops = list()
                    for i in range(num_return_values):
                        dict_with_singles = FlattenedDataOp()
                        for key in call_params.keys():
                            dict_with_singles[key] = ops[key][i]
                        un_split_ops.append(dict_with_singles)
                    ops = tuple(un_split_ops)

                # No splitting to do: Pass everything as-is.
                else:
                    ops = op_rec_column.graph_fn(*call_params)
            else:
                ops = op_rec_column.graph_fn(*flattened_ops)
        # Just pass in everything as-is.
        else:
            ops = op_rec_column.graph_fn(*in_ops)

        # Make sure everything coming from a computation is always a tuple (for out-Socket indexing).
        ops = force_tuple(ops)

        # Always un-flatten all return values. Otherwise, we would allow Dict Spaces
        # with '/' keys in them, which is not allowed.
        ops = op_rec_column.unflatten_output_ops(*ops)

        out_graph_fn_column = op_rec_column.out_graph_fn_column

        # Make sure the number of returned ops matches the number of op-records in the next column.
        assert len(ops) == len(out_graph_fn_column.op_records),\
            "ERROR: Number of returned values of graph_fn '{}/{}' ({}) does not match the number of op-records ({}) " \
            "reserved for the return values of the method!".format(
                op_rec_column.component.name, op_rec_column.graph_fn.__name__, len(ops),
                len(out_graph_fn_column.op_records)
            )
        # Determine the Spaces for each out op and then move it into the respective op and Space slot of the
        # out_graph_fn_column.
        for i, op in enumerate(ops):
            space = get_space_from_op(op)
            # TODO: get already existing parallel column from the component's graph_fn records and
            # compare Spaces with these for match (error if not).
            #if socket.space is not None:
            #    assert space == socket.space,\
            #        "ERROR: Newly calculated output op of graph_fn '{}' has different Space than the Socket that " \
            #        "this op will go into ({} vs {})!".format(graph_fn.name, space, socket.space)
            #else:
            out_graph_fn_column.op_records[i].op = op
            out_graph_fn_column.op_records[i].space = space

    def sanity_check_build(self, component=None):
        # TODO: Obsolete method?
        """
        Checks whether all the `component`'s and sub-components's in-Sockets and graph_fns are input-complete and
        raises detailed error messages if not. Input-completeness means that ..
        a) all in-Sockets of a Component or
        b) all connected incoming Sockets to a GraphFunction
        .. have their `self.space` field defined (is not None).

        Args:
            component (Component): The Component to analyze for input-completeness.
        """
        component = component or self.core_component

        if self.logger.level <= logging.INFO:
            component_print_out(component)

        # Check all the component's graph_fns for input-completeness.
        for graph_fn in component.graph_fns:
            if graph_fn.input_complete is False:
                # Look for the missing in-Socket and raise an Error.
                for in_sock_name, in_sock_record in graph_fn.input_sockets.items():
                    if len(in_sock_record["socket"].op_records) == 0 and \
                            in_sock_name not in component.unconnected_sockets_in_meta_graph:
                        raise YARLError("in-Socket '{}' of GraphFunction '{}' of Component '{}' does not have "
                                        "any incoming ops!".format(in_sock_name, graph_fn.name,
                                                                   component.global_scope))

        # Check component's sub-components for input-completeness (recursively).
        for sub_component in component.sub_components.values():  # type: Component
            if sub_component.input_complete is False:
                # Look for the missing Socket and raise an Error.
                for in_sock in sub_component.input_sockets:
                    if in_sock.space is None:
                        raise YARLError("Component '{}' is not input-complete. In-Socket '{}' does not " \
                                        "have any incoming connections.".
                                        format(sub_component.global_scope, in_sock.name))

            # Recursively call this method on all the sub-component's sub-components.
            self.sanity_check_build(sub_component)

    def get_execution_inputs(self, api_method, *params):
        """
        Creates a fetch-dict and a feed-dict for a graph session call.

        Args:
            api_method (str): The name of the API-method (of the core_component) to execute.
            *params (any): The parameters to pass into the API-method.

        Returns:
            tuple: fetch-dict, feed-dict with relevant args.
        """
        fetch_list = [op_rec.op for op_rec in self.api[api_method][1]]
        feed_dict = dict()
        for i, param in enumerate(params):
            placeholder = self.api[api_method][0][i].op
            feed_dict[placeholder] = param

        return fetch_list, feed_dict

    def set_core_component(self, core_component):
        """
        Sets the core Component that will contain all others.

        Args:
            core_component (Component): The component to set as core.

        Returns:
            Component: The new core Component.
        """
        # Unset current core.
        if self.core_component is not None:
            self.core_component.is_core = False
        # Set new core and return it.
        core_component.is_core = True
        self.core_component = core_component
        return self.core_component
