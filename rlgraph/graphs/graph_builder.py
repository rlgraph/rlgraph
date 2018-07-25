# Copyright 2018 The RLgraph authors. All Rights Reserved.
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
import re
import time

from rlgraph import RLGraphError, Specifiable, get_backend
from rlgraph.components import Component
from rlgraph.spaces import Space, Dict
from rlgraph.spaces.space_utils import get_space_from_op
from rlgraph.utils.input_parsing import parse_summary_spec
from rlgraph.utils.util import force_list, force_tuple, get_shape
from rlgraph.utils.ops import DataOpTuple, FlattenedDataOp, DataOpRecord, DataOpRecordColumnIntoGraphFn
from rlgraph.utils.component_printout import component_print_out

if get_backend() == "tf":
    import tensorflow as tf


class GraphBuilder(Specifiable):
    """
    The graph builder assembles the RLGraph meta-graph by tracing through
    components, sockets and connections and creating the underlying computation
    graph.
    """
    def __init__(self, name="model", action_space=None, summary_spec=None, core_component=None):
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
        # Device specifications.
        self.device_strategy = None
        self.default_device = None
        self.device_map = None

        # Some build stats:
        # Number of meta op-records.
        self.num_meta_ops = 0
        # Number of actual backend ops.
        self.num_ops = 0
        # Number of trainable variables (optimizable weights).
        self.num_trainable_parameters = 0

        # Create an empty core Component into which everything will be assembled by an Algo.
        self.core_component = core_component

        # Maps API method names to in- (placeholders) and out op columns (ops to pull).
        self.api = dict()

        # Dict of unprocessed (and complete) op-record columns by key=op-column ID.
        # Columns that have been forwarded will be erased again from this collection.
        self.unprocessed_in_op_columns = OrderedDict()
        self.unprocessed_complete_in_op_columns = OrderedDict()

    def build_meta_graph(self, input_spaces=None):
        """
        Builds the meta-graph by constructing op-record columns going into and coming out of all API-methods
        and graph_fns.

        Args:
            input_spaces (Optional[Space]): Input spaces for api methods.
        """
        # Time the meta-graph build:
        time_start = time.monotonic()

        # Sanity check input_spaces dict.
        if input_spaces is not None:
            for api_method_name in input_spaces.keys():
                if api_method_name not in self.core_component.api_methods:
                    raise RLGraphError("ERROR: `input_spaces` contains API-method ('{}') that's not defined in "
                                    "core-component '{}'!".format(api_method_name, self.core_component.name))

        # Call all API methods of the core and thereby, create empty in-op columns that serve as placeholders
        # and bi-directional links for the build time.
        for api_method_name, api_method_rec in self.core_component.api_methods.items():
            self.logger.debug("Building meta-graph of API-method '{}'.".format(api_method_name))
            # Create an new in column and map it to the resulting out column.
            in_ops_records = list()
            if input_spaces is not None and api_method_name in input_spaces:
                for i in range(len(force_list(input_spaces[api_method_name]))):
                    in_ops_records.append(DataOpRecord(position=i))
            # Do the actual core API-method call (thereby assembling the meta-graph).
            self.core_component.call(api_method_rec.method, *in_ops_records, ok_to_call_own_api=True)

            # Register core's interface.
            self.api[api_method_name] = (in_ops_records, api_method_rec.out_op_columns[0].op_records)

            # Tag very last out-op-records with op="done", so we know in the build process that we are done.
            for op_rec in api_method_rec.out_op_columns[0].op_records:
                op_rec.is_terminal_op = True

        time_build = time.monotonic() - time_start
        self.logger.info("Meta-graph build completed in {} s.".format(time_build))

        # Get some stats on the graph and report.
        self.num_meta_ops = DataOpRecord._ID + 1
        self.logger.info("\tMeta-graph op-records generated: {}".format(self.num_meta_ops))

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
                raise RLGraphError("Component '{}' has in-Socket ({}) without any incoming connections! If this is "
                                "intended before the build process, you have to add the Socket's name to the "
                                "Component's `unconnected_sockets_in_meta_graph` set. Then this error will be "
                                "suppressed for this Component.".format(component.name, in_sock.name))

        # Check all the component's graph_fns for input-completeness.
        for graph_fn in component.graph_fns:  # type: GraphFunction
            for in_sock_rec in graph_fn.input_sockets.values():
                in_sock = in_sock_rec["socket"]
                if len(in_sock.incoming_connections) == 0 and \
                        in_sock.name not in component.unconnected_sockets_in_meta_graph:
                    raise RLGraphError("GraphFn {}/{} has in-Socket ({}) without any incoming "
                                    "connections!".format(component.name, graph_fn.name, in_sock_rec["socket"].name))

        # Recursively call this method on all the sub-component's sub-components.
        for sub_component in component.sub_components.values():
            #self.build_steps += 1
            #if self.build_steps >= self.MAX_ITERATIVE_LOOPS:
            #    raise RLGraphError("Error sanity checking graph, reached max recursion steps: {}".format(
            #        self.MAX_ITERATIVE_LOOPS
            #    ))
            self.sanity_check_meta_graph(sub_component)

    def build_graph(self, input_spaces, available_devices,
                    device_strategy="default", default_device=None, device_map=None):
        """
        The actual iterative depth-first search algorithm to build our graph from the already existing
        meta-Graph structure.
        Starts from a set of DataOpRecords populated with the initial placeholders (from input
        Spaces). Keeps pushing these ops through the meta-graph until a non-complete graph_fn
        or a non-complete Component (with at least one non-complete API-method) is reached.
        Replaces the ops in the set with the newly reached ones and re-iterates like this until all op-records
        in the entire meta-graph have been filled with actual ops.

        Args:
            input_spaces (dict):
            available_devices (list): Devices which can be used to assign parts of the graph
                during graph assembly.
            device_strategy (Optional[str]): Device strategy.
            default_device (Optional[str]): Default device identifier.
            device_map (Optional[Dict]): Dict of Component names mapped to device names to place the Component's ops.
        """
        # Time the build procedure.
        time_start = time.monotonic()

        # Set devices usable for this graph.
        self.available_devices = available_devices
        self.device_strategy = device_strategy
        self.default_device = default_device
        self.device_map = device_map or dict()

        # Create the first actual ops based on the input-spaces.
        # Some ops can only be created later when variable-based-Spaces are known (op_records_to_process_later).
        op_records_to_process, op_records_to_process_later = self.build_input_space_ops(input_spaces)

        # Collect all components and add those op-recs to the set that are constant.
        components = self.get_all_components()
        for component in components:
            op_records_to_process.update(component.constant_op_records)
            # Check whether the Component is input-complete (and build already if it is).
            self.build_component_when_input_complete(component, op_records_to_process)

        op_records_list = sorted(op_records_to_process, key=lambda rec: rec.id)

        # Re-iterate until our bag of op-recs to process is empty.
        loop_counter = 0
        while len(op_records_list) > 0:
            new_op_records_to_process = set()
            for op_rec in op_records_list:  # type: DataOpRecord
                # There are next records:
                if len(op_rec.next) > 0:
                    # Push the op-record forward one step.
                    for next_op_rec in sorted(op_rec.next, key=lambda rec: rec.id):  # type: DataOpRecord
                        # If not last op in this API-method -> continue.
                        if next_op_rec.is_terminal_op is False:
                            assert next_op_rec.op is None
                            new_op_records_to_process.add(next_op_rec)
                        # Push op and Space into next op-record.
                        next_op_rec.op = op_rec.op
                        next_op_rec.space = op_rec.space

                        # Did we enter a new Component? If yes, check input-completeness and
                        # - If op_rec.column is None -> We are at the very beginning of the graph (op_rec.op is a
                        # placeholder).
                        if op_rec.column is None or op_rec.column.component is not next_op_rec.column.component:
                            self.build_component_when_input_complete(next_op_rec.column.component,
                                                                     new_op_records_to_process)

                # No next records:
                # - Op belongs to a column going into a graph_fn.
                elif isinstance(op_rec.column, DataOpRecordColumnIntoGraphFn):
                    # If column complete AND has not been sent through the graph_fn yet -> Call the graph_fn.
                    if op_rec.column.is_complete() and op_rec.column.already_sent is False:
                        # Only call the graph_fn if the Component is already input-complete.
                        if op_rec.column.component.input_complete:
                            # Call the graph_fn with the given column and call-options.
                            self.run_through_graph_fn_with_device_and_scope(op_rec.column)
                            # Store all resulting op_recs (returned by the graph_fn) to be processed next.
                            new_op_records_to_process.update(op_rec.column.out_graph_fn_column.op_records)
                        # Component not input-complete. Keep coming back with this op.
                        else:
                            self.build_component_when_input_complete(op_rec.column.component, new_op_records_to_process)
                            new_op_records_to_process.add(op_rec)
                    # - Op column is not complete yet: Discard this one (as others will keep coming in anyway).
                    # - Op column has already been sent (sibling ops may have arrive in same iteration).
                # - Op belongs to a column coming from a graph_fn or an API-method, but the op is no longer used.
                # -> Ignore Op.

            # Sanity check, whether we are stuck.
            new_op_records_list = sorted(new_op_records_to_process, key=lambda rec: rec.id)
            if op_records_list == new_op_records_list:
                raise RLGraphError("Build procedure is deadlocked. Most likely, you are having a "
                                "circularly dependent Component in your meta-graph. The current op-records to process "
                                "are: {}".format(new_op_records_list))

            op_records_list = new_op_records_list

            # If we are done with the build, check for API-methods' ops that are dependent on variables
            # generated during the build and build these now.
            if len(op_records_list) == 0 and op_records_to_process_later is not None:
                op_records_list = list(op_records_to_process_later)
                # Invalidate later-set.
                op_records_to_process_later = None

                # Loop through the op_records list and sanity check for "variables"-dependent Spaces, then get these
                # Spaces, create the placeholders and keep building.
                for op_rec in op_records_list:
                    space_desc = op_rec.space
                    mo = re.search(r'^variables:(.+)', space_desc)
                    assert mo
                    component_path = mo.group(1).split("/")
                    component = self.core_component
                    for level in component_path:
                        assert level in component.sub_components,\
                            "ERROR: `component_path` ('{}') contains non-existent Components!".format(component_path)
                        component = component.sub_components[level]
                    var_spaces = {key: get_space_from_op(value) for key, value in sorted(
                        component.get_variables(custom_scope_separator="-").items()
                    )}
                    var_space = Dict(var_spaces)
                    op_rec.space = var_space
                    op_rec.op = self.get_placeholder("api-", space=var_space, component=self.core_component)

            loop_counter += 1

        time_build = time.monotonic() - time_start
        self.logger.info("Computation-Graph build completed in {} s ({} iterations).".format(time_build, loop_counter))

        # Get some stats on the graph and report.
        self.num_ops = self.count_ops()
        self.logger.info("\tBackend ops generated: {}".format(self.num_ops))

        self.num_trainable_parameters = self.count_trainable_parameters()
        self.logger.info("\tNumber of trainable parameters: {}".format(self.num_trainable_parameters))

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
        op_records_to_process_later = set()

        for api_method_name, (in_op_records, _) in self.api.items():
            spaces = force_list(input_spaces[api_method_name]) if input_spaces is not None and \
                                                                  api_method_name in input_spaces else list()
            assert len(spaces) == len(in_op_records)

            # Create the placeholder and store it in the given DataOpRecords.
            for i, space in enumerate(spaces):
                # Space is dependent on the variables of some sub-component (wait with the construction of the
                # placeholder until after the build).
                if isinstance(space, str) and re.match(r'^variables:', space):
                    in_op_records[i].space = space
                    op_records_to_process_later.add(in_op_records[i])
                    continue

                # Construct Space from a spec.
                elif not isinstance(space, Space):
                    space = Space.from_spec(space)

                in_op_records[i].space = space
                in_op_records[i].op = self.get_placeholder(
                    name="api-" + api_method_name + "/param-" + str(i),
                    space=space,
                    component=next(iter(in_op_records[0].next)).column.component
                )
                op_records_to_process.add(in_op_records[i])

        return op_records_to_process, op_records_to_process_later

    def get_placeholder(self, name, space, component):
        """
        Generates one or more placeholders given a name, space and a component (for device inference).

        Args:
            name (str): The name of the placeholder to create.
            space (Space): The Space object to generate the placeholder for.
            component (Component): The Component into which the placeholder will go (needed  for automatic device
                inference).

        Returns:
            DataOp: The generated placeholder(s) as a DataOp (e.g. DataOpTuple, SingleDataOp, etc..).
        """
        device = self.get_device(component, variables=True)
        placeholder = None
        if get_backend() == "tf":
            with tf.device(device):
                placeholder = space.get_tensor_variable(name=name, is_input_feed=True)
        return placeholder

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
                ret.extend(sorted(list_[l], key=lambda c: c.scope))
            return ret

    def build_component_when_input_complete(self, component, op_records_to_process):
        # Not input complete yet -> Check now.
        if component.input_complete is False:
            spaces_dict = component.check_input_completeness()
            # call `when_input_complete` once on that Component.
            if spaces_dict is not None:
                self.logger.debug("Component {} is input-complete; spaces_dict={}".
                                  format(component.name, spaces_dict))
                device = self.get_device(component, variables=True)
                component.when_input_complete(spaces_dict, self.action_space, device)
                # Call all no-input graph_fns of the new Component.
                for no_in_col in component.no_input_graph_fn_columns:
                    # Do not call _variables (only later, when Component is also variable-complete).
                    if no_in_col.graph_fn.__name__ != "_graph_fn__variables":
                        self.run_through_graph_fn_with_device_and_scope(no_in_col)
                        # Keep working with the generated output ops.
                        op_records_to_process.update(no_in_col.out_graph_fn_column.op_records)

        # Check variable-completeness and actually call the _variable graph_fn if the component just became
        # "variable-complete".
        if component.input_complete is True and component.variable_complete is False and \
                component.check_variable_completeness():
            # The graph_fn _variables has some in-op-columns that need to be run through the function.
            if "_graph_fn__variables" in component.graph_fns:
                graph_fn_rec = component.graph_fns["_graph_fn__variables"]
                # TODO: Think about only running through no-input-graph-fn once no matter how many in-op-columns it has.
                # TODO: Then link the first in-op-column (empty) to all out-op-columns.
                for i, in_op_col in enumerate(graph_fn_rec.in_op_columns):
                    self.run_through_graph_fn_with_device_and_scope(in_op_col)
                    # Keep working with the generated output ops.
                    op_records_to_process.update(graph_fn_rec.out_op_columns[i].op_records)

    def run_through_graph_fn_with_device_and_scope(self, op_rec_column):
        """
        Runs through a graph_fn with the given ops and thereby assigns a device (Component's device or GraphBuilder's
        default) to the ops generated by a graph_fn.

        Args:
            op_rec_column (DataOpRecordColumnIntoGraphFn): The column of DataOpRecords to be fed through the
                graph_fn.
        """
        # Get the device for the ops generated in the graph_fn (None for custom device-definitions within the graph_fn).
        device = self.get_device(op_rec_column.component, variables=False)

        if get_backend() == "tf":
            # Assign proper device to all ops created in this context manager.
            with tf.device(device):
                # Name ops correctly according to our Component hierarchy.
                with tf.name_scope(op_rec_column.component.global_scope+
                                   ('/' if op_rec_column.component.global_scope else "")):
                    self.logger.debug(
                        "Assigning device '{}' to graph_fn '{}' (scope '{}').".
                        format(device, op_rec_column.graph_fn.__name__, op_rec_column.component.global_scope)
                    )
                    self.run_through_graph_fn(op_rec_column)

        # Tag column as already sent through graph_fn.
        op_rec_column.already_sent = True

        # Store assigned names for debugging.
        if device is not None:
            if device not in self.device_component_assignments:
                self.device_component_assignments[device] = [str(op_rec_column.graph_fn.__name__)]
            else:
                self.device_component_assignments[device].append(str(op_rec_column.graph_fn.__name__))

    def get_device(self, component, variables=False):
        """
        Determines and returns a device based on a given Component (or `self.default_device`).
        Also does some sanity checking against our `device_strategy`.

        Args:
            component (Component): The Component to check for a defined device.
            variables (bool): Whether the device is for the variables of the Component (vs the ops).

        Returns:
            str: The device to use for the component (its ops or variables or both).
        """
        # Component specifies it's own device: Use that.
        # Then follow our device_map.
        # Last resort: Use `self.default_device` (may be None).
        device = component.device
        # Try a map lookup via global-scope of the component.
        if device is None:
            # Sort by scope-length (such that the most specific assignments have priority).
            for key in sorted(self.device_map.keys(), key=len, reverse=True):
                if re.search(r'^{}\b'.format(key), component.global_scope):
                    device = self.device_map[key]
                    break
            else:
                device = self.default_device
        # Device is specific to whether we are creating variables or ops.
        if isinstance(device, dict):
            device = device["variables"] if variables is True else device["ops"]

        # If device is not available, use the default device (or None).
        if device is not None and device not in self.available_devices:
            device = self.device_map.get(component.name, self.default_device)
        # Device is specific to whether we are creating variables or ops.
        if isinstance(device, dict):
            device = device["variables"] if variables is True else device["ops"]

        return device

    def get_subgraph(self, api_methods):
        """
        Computes and returns the sub-graph necessary to compute provided API methods.

        Args:
            api_methods Union[list,str]: Api method(s) considered for the subgraph.

        Returns:
            Component: Container component holding the subgraph.
        """
        if isinstance(api_methods, str):
            api_methods = [api_methods]

        subgraph_container = Component(scope="sub_graph")
        subgraph_components = set()

        # For every api method, trace forward from inputs until all relevant components identified.
        for api_method in api_methods:
            # Start with the input set for this method.
            api_input_records = self.core_component[api_method].in_op_columns[0].op_records
            # OBSOLETE: use "is_terminal_op is True" (in DataOpRecord).
            # out_columns = self.core_component[api_method].out_op_columns
            # for column in out_columns:
            #    for out_op_rec in column.out_op_records:
            #        out_op_rec.op = "done"
            op_records_list = sorted(api_input_records, key=lambda rec: rec.id)

            # Re-iterate until our bag of op-recs to process is empty.
            loop_counter = 0
            while len(op_records_list) > 0:
                new_op_records_to_process = set()
                for op_rec in op_records_list:  # type: DataOpRecord
                    # There are next records:
                    if len(op_rec.next) > 0:
                        # Push the op-record forward one step.
                        for next_op_rec in sorted(op_rec.next, key=lambda rec: rec.id):  # type: DataOpRecord
                            # If not last op in this API-method ("done") -> continue.
                            # Otherwise, replace "done" with actual op.
                            if next_op_rec.op != "done":
                                assert next_op_rec.op is None

                            # Did we enter a new Component? If yes, add current component.
                            # - If op_rec.column is None -> We are at the very beginning of the graph (op_rec.op is a
                            # placeholder).
                            if op_rec.column is None or op_rec.column.component is not next_op_rec.column.component:
                                # Add this component to the sub-graph.
                                if op_rec.column.component not in subgraph_components:
                                    subgraph_container.add_components(op_rec.column.component)

                    # No next records:
                    # - Op belongs to a column going into a graph_fn.
                    elif isinstance(op_rec.column, DataOpRecordColumnIntoGraphFn):
                        new_op_records_to_process.update(op_rec.column.out_graph_fn_column.op_records)

                # No sanity checks because meta graph was already built successfully.
                new_op_records_list = sorted(new_op_records_to_process, key=lambda rec: rec.id)
                op_records_list = new_op_records_list
                loop_counter += 1
        return subgraph_container

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
                            raise RLGraphError("Different split-runs through {} do not return the same number of "
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
            # Make sure the receiving op-record is still empty.
            assert out_graph_fn_column.op_records[i].op is None
            out_graph_fn_column.op_records[i].op = op
            out_graph_fn_column.op_records[i].space = space

    @staticmethod
    def count_trainable_parameters():
        """
        Counts the number of trainable parameters (e.g. tf.Variables) to get a rough idea of how complex
        our Model is.

        Returns:
            int: The number of trainable parameters in the graph.
        """
        num_trainable_parameters = 0
        if get_backend() == "tf":
            for variable in tf.trainable_variables():
                # shape is an array of tf.Dimension
                num_trainable_parameters += get_shape(variable, flat=True)

        return num_trainable_parameters

    @staticmethod
    def count_ops():
        """
        Counts the number of all backend-specific ops present in the graph.
        This includes variables and placeholders.

        Returns:
            int: The number of backend-specific ops in the graph.
        """
        if get_backend() == "tf":
            return len(tf.get_default_graph().as_graph_def().node)
        return 0

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
                        raise RLGraphError("in-Socket '{}' of GraphFunction '{}' of Component '{}' does not have "
                                        "any incoming ops!".format(in_sock_name, graph_fn.name,
                                                                   component.global_scope))

        # Check component's sub-components for input-completeness (recursively).
        for sub_component in component.sub_components.values():  # type: Component
            if sub_component.input_complete is False:
                # Look for the missing Socket and raise an Error.
                for in_sock in sub_component.input_sockets:
                    if in_sock.space is None:
                        raise RLGraphError("Component '{}' is not input-complete. In-Socket '{}' does not " \
                                        "have any incoming connections.".
                                        format(sub_component.global_scope, in_sock.name))

            # Recursively call this method on all the sub-component's sub-components.
            self.sanity_check_build(sub_component)

    def get_execution_inputs(self, *api_methods):
        """
        Creates a fetch-dict and a feed-dict for a graph session call.

        Args:
            api_methods (dict): See `rlgraph.graphs.graph_executor` for details.

        Returns:
            Tuple[list,dict]: Fetch-list, feed-dict with relevant args.
        """
        fetch_dict = dict()
        feed_dict = dict()

        for api_method in api_methods:
            if api_method is None:
                continue

            params = list()
            return_ops = None
            if isinstance(api_method, (list, tuple)):
                params = force_list(api_method[1])
                return_ops = force_list(api_method[2]) if len(api_method) > 2 else None
                api_method = api_method[0]

            if api_method not in self.api:
                raise RLGraphError("No API-method with name '{}' found!".format(api_method))

            if return_ops is None:
                fetch_dict[api_method] = [op_rec.op for i, op_rec in enumerate(self.api[api_method][1])]
            else:
                fetch_dict[api_method] = [self.api[api_method][1][i].op for i in return_ops]

            for i, param in enumerate(params):
                if len(self.api[api_method][0]) <= i:
                    raise RLGraphError("API-method with name '{}' only has {} input parameters! You passed in "
                                    "{}.".format(api_method, len(self.api[api_method][0]), len(params)))
                placeholder = self.api[api_method][0][i].op  # 0=input op-recs; i=ith input op-rec
                if isinstance(placeholder, DataOpTuple):
                    for ph, p in zip(placeholder, param):
                        feed_dict[ph] = p
                else:
                    feed_dict[placeholder] = param

        return fetch_dict, feed_dict

