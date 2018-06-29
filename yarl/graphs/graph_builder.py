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
import itertools
import logging
import numpy as np
import re

from yarl import YARLError, Specifiable, get_backend
from yarl.components import Component
from yarl.spaces import Space
from yarl.spaces.space_utils import split_flattened_input_ops, get_space_from_op
from yarl.utils.input_parsing import parse_summary_spec
from yarl.utils.util import all_combinations, force_list, force_tuple, get_shape
from yarl.utils.ops import SingleDataOp, FlattenedDataOp, DataOpRecord, DataOpRecordColumn
from yarl.utils.component_printout import component_print_out

if get_backend() == "tf":
    import tensorflow as tf


class GraphBuilder(Specifiable):
    """
    The graph builder assembles the YARL meta-graph by tracing through
    components, sockets and connections and creating the underlying computation
    graph.
    """
    # Break graph building if caught in a loop.
    MAX_RECURSIVE_CALLS = 100

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

        # Counting recursive steps.
        self.build_steps= 0

        # Create an empty core Component into which everything will be assembled by an Algo.
        self.core_component = None  # Component(name=self.name, is_core=True)
        # A dict used for lookup of all combinations that are possible for a given set of given in-Socket
        # names (inside a call to `self.call`).
        #self.input_combinations = dict()

        # Maps API method names to in- (placeholders) and out op columns (ops to pull).
        self.api = dict()

        # Dict of op-record columns by key=op-column ID.
        #self.in_op_columns = OrderedDict()
        # Dict of unprocessed (and complete) op-record columns by key=op-column ID.
        # Columns that have been forwarded will be erased again from this collection.
        self.unprocessed_in_op_columns = OrderedDict()
        self.unprocessed_complete_in_op_columns = OrderedDict()

        # Some registries that we need in order to build the Graph from core:
        # key=DataOpRecord; value=set of required DataOpRecords OR leftmost in-Sockets to calculate the key's op.
        #self.op_record_registry = dict()
        # Only for core's in-Sockets.
        # key=in-Socket name; value=DataOp (e.g. tf.placeholder) that goes into this socket.
        #self.in_socket_registry = dict()
        # key=out-Socket name; value=set of necessary in-Socket names that we need in order to calculate
        #   the out-Socket's op output.
        #self.out_socket_registry = dict()
        # Maps an out-Socket name+in-Socket/Space-combination to an actual DataOp to fetch from our Graph.
        #self.call_registry = dict()  # key=(FixMe: documentation)

    def build_graph(self, input_spaces, available_devices, default_device):
        """
        Builds the actual backend-specific graph from the YARL metagraph.
        Loops through all our sub-components starting at core and assembles the graph by creating placeholders,
        following Socket->Socket connections and running through our GraphFunctions to generate DataOps.

        Args:
            input_spaces (dict):
            available_devices (list): Devices which can be used to assign parts of the graph
                during graph assembly.
            default_device (str): Default device identifier.
        """
        # Build the meta-graph and collect all empty in-op-columns.
        self.build_meta_graph(input_spaces)
        for api_method_rec in self.core_component.api_methods.values():
            for in_op_col in api_method_rec.in_op_columns:
                self.collect_op_column(in_op_col)

        # Before we start, sanity check the meta graph for obvious flaws.
        #self.sanity_check_meta_graph()

        # Set devices usable for this graph.
        self.available_devices = available_devices
        self.default_device = default_device

        # Collect all components and store their input-completeness state.
        # TODO: remove hard-coding...
        #input_completeness = dict({self.core_component: False,
        #                           self.core_component.sub_components["dummy-with-sub-components"]: False})

        # Actually build the graph.
        # Push all spaces through the API methods, then enter the main iterative DFS while loop.
        for method_name, api_method_rec in self.core_component.api_methods.items():
            self.push_spaces_into_api_method(input_spaces[method_name], api_method_rec)

        # While some sub-components are still incomplete.
        current_scope = self.core_component.global_scope
        while not all(input_completeness.values()):
            # Components to do on this level:
            components = [
                c for c in input_completeness.keys() if re.search(r'^{}'.format(current_scope), c.global_scope)
            ]
            for component in components:
                was_input_complete = component.input_complete
                space_dict = component.check_input_completeness()
                # State changed -> Iterate through this component.
                if space_dict is not None and was_input_complete is False:
                    input_completeness[component] = True  # Change entry
                    self.run_through_component(component, space_dict)

        #space_dict = self.core_component.check_input_completeness()
        #assert space_dict is not None
        #self.build_component(self.core_component, space_dict)

        # Check whether all our components and graph_fns are now input-complete.
        self.sanity_check_build()

        # Memoize possible input combinations for out-Socket `execution` calls.
        self.memoize_inputs()

        # Registers actual ops with the different out-Sockets, so we know, which ops to execute for a given
        # out-Socket/input-feed-data combination.
        self.register_ops()

    def build_meta_graph(self, input_spaces):
        # Call all API methods of the core and thereby, create empty in-op columns that serve as placeholders
        # and directed links for the build time.
        for method_name, api_method_rec in self.core_component.api_methods.items():
            # Create an new in column and map it to the resulting out column.
            in_ops_records = [DataOpRecord()] * len(force_list(input_spaces[method_name]))
            self.core_component.call(api_method_rec.method, *in_ops_records)
            #out_op_col = DataOpRecordColumn(op_records=outs)
            # Register interface.
            self.api[method_name] = (in_ops_records, api_method_rec.out_op_columns[0].op_records)

    def collect_op_column(self, column):
        new_columns = set()
        for in_op_col in column:
            self.unprocessed_in_op_columns[in_op_col.id] = in_op_col
            if in_op_col.is_complete():
                self.unprocessed_complete_in_op_columns[in_op_col.id] = in_op_col
            # Follow each single op and collect new columns.
            for op_rec in in_op_col.op_records:
                # Follow each next op-column:
                for next_op_rec in op_rec.next:
                    # Completely new column: Process it.
                    if next_op_rec[0] not in self.unprocessed_in_op_columns:
                        # 0=column; 1=op-rec slot in that column
                        new_columns.add(next_op_rec[0])

        # Recurse:
        for op_col in new_columns:
            self.collect_op_column(op_col)

    def sanity_check_meta_graph(self, component=None):
        """
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
            if self.build_steps >= self.MAX_RECURSIVE_CALLS:
                raise YARLError("Error sanity checking graph, reached max recursion steps: {}".format(
                    self.MAX_RECURSIVE_CALLS
                ))
            self.sanity_check_meta_graph(sub_component)

    def push_spaces_into_api_method(self, spaces, api_method_rec):
        """
        Stores Spaces information for an APIMethodRecord and creates ops and op-records from the given Spaces.
        The APIMethodRecord must be one of the core Component's.

        Args:
            spaces (List[Spaces,dict]: List of Space objects or specification dicts to create the Spaces that go into
                the given APIMethodRecord.
            api_method_rec (APIMethodRecord): The APIMethodRecord object to populate with op-records and Spaces.
        """
        assert api_method_rec.component.is_core,\
            "ERROR: Can only push a Space into a core's API-method (method={})!".format(api_method_rec.method.__name__)

        # Create the placeholder and wrap it in a DataOpRecord, then add the DataOpRecords as a column to the
        # APIMethodRecord.
        api_method_rec.spaces = list()
        column = list()
        for i, space in enumerate(force_list(spaces)):
            if not isinstance(space, Space):
                space = Space.from_spec(space)
            api_method_rec.spaces.append(space)
            op = space.get_tensor_variable(name=api_method_rec.method.__name__+"/"+str(i), is_input_feed=True)
            op_rec = DataOpRecord(op=op)
            column.append(op_rec)
        api_method_rec.in_op_columns.append(column)

        #op_rec = DataOpRecord(op)
        #socket.op_records.add(op_rec)

        # Keep track of which input op (e.g. tf.placeholder) goes into this Socket.
        #self.in_socket_registry[socket.name] = op

        # Remember, that this DataOp goes into a Socket at the very beginning of the Graph (e.g. a
        # tf.placeholder).
        #self.op_record_registry[op_rec] = {socket}

    def run_through_component(self, component):
        assert component.input_complete

        for api_method_rec in component.api_methods:
            # Loop through all op columns separately.
            for in_op_col in api_method_rec.in_op_columns:
                # Skip op columns that are already complete.
                if self.check_op_column_complete(in_op_col):
                    continue
                # This op-column is not complete yet.


    def build_component(self, component, input_spaces):
        """
        Called when a Component has all its incoming Spaces known. Only then can we sanity check the input
        Spaces and create the Component's variables.

        Args:
            component (Component): The Component that now has all its input Spaces defined.
            input_spaces (dict): A dict mapping all API method names of `component` to a tuple of Space objects that
                go into the API method.
        """
        assert component.input_complete is True, "ERROR: Component {} is not input complete!".format(component.name)
        self.logger.debug("Component {} is input-complete; space-dict={}".format(component.name, input_spaces))
        # Component is complete now, allow it to sanity check its api_methods and create its variables.
        component.when_input_complete(input_spaces, self.action_space, self.summary_spec["summaries_regexp"])

        # Call each method in input_spaces.
        for method_name in input_spaces.keys():
            api_method_rec = component.api_methods[method_name]
            self.push_api_method_rec(api_method_rec)

    @staticmethod
    def check_op_column_complete(op_column):
        for op_rec in op_column:
            if op_rec.op is None:
                return False
        else:
            return True

    def push_api_method_rec(self, api_method_rec):
        # Check each input op column.
        for in_op_col in api_method_rec.in_op_columns:
            # Check whether this column is complete.
            if self.check_op_column_complete(in_op_col):
                pass
            # Op column is not complete.
            else:
                # Check whether we have a matching
                pass

    def push_from_socket(self, socket):
        # Skip this Socket, if it doesn't have a Space (no incoming connection).
        # Assert that it's ok for the component to leave this Socket open.
        if socket.space is None:
            assert socket.name in socket.component.unconnected_sockets_in_meta_graph
            return

        for outgoing in socket.outgoing_connections:
            # Push Socket into Socket.
            if isinstance(outgoing, Socket):
                #print("SOCK {}/{} -> {}/{}".format(socket.component.name, socket.name, outgoing.component.name, outgoing.name))
                self.push_socket_into_socket(socket, outgoing)
            # Push Socket into GraphFunction.
            elif isinstance(outgoing, GraphFunction):
                self.push_from_graph_fn(outgoing)
            # Error.
            else:
                raise YARLError("ERROR: Outgoing connection ({}) must be of type Socket or GraphFunction!".\
                                format(outgoing))

    def push_socket_into_socket(self, socket, next_socket):
        """
        Pushes op records from one Socket into a next connected one.

        Args:
            socket (Socket): The Socket object to push from.
            next_socket (Socket): The Socket object to push into.
        """
        assert socket.space is not None
        assert next_socket.space is None or socket.space == next_socket.space,\
            "ERROR: Socket '{}' already has Space '{}', but incoming connection '{}' has Space '{}'! " \
            "Incoming Spaces must always be the same.".format(next_socket, next_socket.space, socket, socket.space)

        was_input_complete = next_socket.component.input_complete

        self.logger.debug("Socket {}/{} -> Socket {}/{}".format(socket.component.name, socket.name,
                                                               next_socket.component.name, next_socket.name))
        next_socket.space = socket.space

        # Push the op-records into the next Socket.
        for op_record in socket.op_records:  # type: DataOpRecord
            if op_record.op is not None:
                next_socket.push_op_from_incoming_socket(op_record.op, in_socket_name=socket.name,
                                                         in_op_record=op_record)

        # Continue with the build logic.
        self.after_socket_update(next_socket, was_input_complete)

    def after_socket_update(self, socket, was_input_complete):
        # The Component of the Socket has already been input-complete. Keep pushing the Socket forward.
        if was_input_complete is True:
            self.push_from_socket(socket)
        else:
            # Check again for input-completeness.
            space_dict = socket.component.check_input_completeness()
            # Component has just become input-complete: Build it.
            if socket.component.input_complete:
                self.build_component(socket.component, space_dict)

    def push_from_graph_fn(self, graph_fn):
        """
        Builds outgoing graph function ops using `socket`'s component's device or the GraphBuilder's default
        one.

        Args:
            graph_fn (GraphFunction): Graph function object to build output ops for.
        """
        # Check for input-completeness of this graph_fn.
        if graph_fn.check_input_completeness():
            # We have to specify the device and the variable scope here as we will be running through a
            # GraphFunction, which may add ops to the graph.
            assigned_device = graph_fn.component.device or self.default_device
            self.run_through_graph_fn_with_device_and_scope(graph_fn, assigned_device)

            # Store assigned names for debugging.
            if assigned_device not in self.device_component_assignments:
                self.device_component_assignments[assigned_device] = [str(graph_fn)]
            else:
                self.device_component_assignments[assigned_device].append(str(graph_fn))

            # Keep moving through this graph_fn's out-Sockets (if input-complete).
            if graph_fn.input_complete:
                for slot, out_socket in enumerate(graph_fn.output_sockets):
                    self.push_from_socket(out_socket)  #, graph_fn, slot)

    def run_through_graph_fn_with_device_and_scope(self, graph_fn, assigned_device):
        """
        Assigns device to the ops generated by a graph_fn.

        Args:
            graph_fn (GraphFunction): GraphFunction to assign device to.
            assigned_device (str): Device identifier.
        """
        if get_backend() == "tf":
            if assigned_device not in self.available_devices:
                self.logger.error("Assigned device {} for graph_fn {} not in available devices:\n {}".
                                  format(assigned_device, graph_fn, self.available_devices))

            # Assign proper device to all ops created in this context manager.
            with tf.device(assigned_device):
                # Name ops correctly according to our Component hierarchy.
                with tf.name_scope(graph_fn.component.global_scope+('/' if graph_fn.component.global_scope else "")):
                    self.logger.debug("Assigning device {} to graph_fn {} (scope {}).".
                                      format(assigned_device, graph_fn, graph_fn.component.global_scope))
                    self.run_through_graph_fn(graph_fn)

    def run_through_graph_fn(self, graph_fn):
        """
        Pushes all incoming ops through the method of this GraphFunction object.
        The ops are collected from incoming Sockets and optionally flattened and/or split
        before pushing them through the method and the return values optionally unflattened.

        Args:
            graph_fn (GraphFunction): The GraphFunction object to run through (its method) with all
                possible in-Socket combinations (only those that have not run yet through the method).
        """
        in_op_records = [in_sock_rec["socket"].op_records for in_sock_rec in graph_fn.input_sockets.values()]
        in_op_records_combinations = list(itertools.product(*in_op_records))

        for in_op_record_combination in in_op_records_combinations:
            # Make sure we call the computation method only once per input-op combination.
            if in_op_record_combination in graph_fn.in_out_records_map:
                continue
            # If any of the ops in the combination is not set yet -> skip this combination for now.
            elif any([r.op is None for r in in_op_record_combination]):
                continue
            # Make sure only ops with the same group ID (or no group ID) are pushed through together.
            # Find first non-None group ID, all others must match that one (or be None).
            op_group = [r.group for r in in_op_record_combination if r.group is not None]
            op_group = op_group[0] if len(op_group) > 0 else None
            if any([r.group is not None and r.group != op_group for r in in_op_record_combination]):
                continue

            # Replace constant-value Sockets with their SingleDataOp's constant numpy values
            # and the DataOps with their actual ops (`op` property of DataOp).
            actual_call_params = [
                op_rec.op.constant_value if isinstance(op_rec.op, SingleDataOp) and
                                            op_rec.op.constant_value is not None
                else op_rec.op for op_rec in in_op_record_combination
            ]

            # Build the ops from this input-combination.
            # Flatten input items.
            if graph_fn.flatten_ops is not False:
                flattened_ops = graph_fn.flatten_input_ops(*actual_call_params)
                # Split into SingleDataOps?
                if graph_fn.split_ops:
                    call_params = split_flattened_input_ops(graph_fn.add_auto_key_as_first_param, *flattened_ops)
                    # There is some splitting to do. Call graph_fn many times (one for each split).
                    if isinstance(call_params, FlattenedDataOp):
                        ops = dict()
                        num_return_values = -1
                        for key, params in call_params.items():
                            ops[key] = force_tuple(graph_fn.method(*params))
                            if num_return_values >= 0 and num_return_values != len(ops[key]):
                                raise YARLError("Different split-runs through {} do not return the same number of "
                                                "values!".format(graph_fn.name))
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
                        ops = graph_fn.method(*call_params)
                else:
                    ops = graph_fn.method(*flattened_ops)
            # Just pass in everything as-is.
            else:
                ops = graph_fn.method(*actual_call_params)

            # OBSOLETE: always must un-flatten all return values. Otherwise, we would allow Dict Spaces
            # with '/' keys in them, which is not allowed.
            #if graph_fn.unflatten_ops:
            ops = graph_fn.unflatten_output_ops(*force_tuple(ops))

            # Make sure everything coming from a computation is always a tuple (for out-Socket indexing).
            ops = force_tuple(ops)

            # Make sure the number of returned ops matches the number of outgoing Sockets from thie graph_fn
            assert len(ops) == len(graph_fn.output_sockets),\
                "ERROR: Number of returned values of graph_fn '{}/{}' ({}) does not match number of out-Sockets ({}) " \
                "of this GraphFunction!".format(graph_fn.component.name, graph_fn.name, len(ops),
                                                len(graph_fn.output_sockets))

            # ops are now the raw graph_fn output: Need to convert it back to records.
            op_records = list()

            # Move graph_fn results into next Socket(s).
            for i, socket in enumerate(graph_fn.output_sockets):
                self.logger.debug("GraphFn {}/{} -> return-slot {} -> {} -> Socket {}/{}".format(
                    graph_fn.component.name, graph_fn.name, i, ops, socket.component.name, socket.name)
                )
                # Store op_rec in the respective outgoing Socket (and make sure Spaces match).
                space = get_space_from_op(ops[i])
                if socket.space is not None:
                    assert space == socket.space,\
                        "ERROR: Newly calculated output op of graph_fn '{}' has different Space than the Socket that " \
                        "this op will go into ({} vs {})!".format(graph_fn.name, space, socket.space)
                else:
                    socket.space = space

                # Add the new op to the Socket (or place into an existing (empty) op_record that matches the op_group).
                op_rec = socket.push_op_from_incoming_graph_fn(op=ops[i], op_group=op_group)
                self.op_record_registry[op_rec] = set(in_op_record_combination)

                # Make sure all op_records do not contain SingleDataOps with constant_values. Any
                # in-Socket-connected constant values need to be converted to actual ops during a graph_fn call.
                assert not isinstance(op_rec.op, SingleDataOp), \
                    "ERROR: graph_fn '{}' returned a SingleDataOp with constant_value set to '{}'! " \
                    "This is not allowed. All graph_fns must return actual (non-constant) ops.". \
                    format(graph_fn.name, op_rec.op.constant_value)

                op_records.append(op_rec)

            graph_fn.in_out_records_map[in_op_record_combination] = tuple(op_records)

    def memoize_inputs(self):
        # Memoize possible input-combinations (from all our in-Sockets)
        # so we don't have to do this every time we get a call to `self.execute`.
        in_names = sorted(list(map(lambda s: s.name, self.core_component.input_sockets)))
        input_combinations = all_combinations(in_names, descending_length=True)
        # Store each combination and its sub-combinations in self.input_combinations.
        for input_combination in input_combinations:
            self.input_combinations[tuple(input_combination)] = \
                all_combinations(input_combination, descending_length=True)

    def register_ops(self):
        # Now use the ready op/socket registries to determine for which out-Socket we need which api_methods.
        # Then we will be able to derive the correct op for any given (out-Socket+in-Socket+in-shape)-combination
        # passed into the call method.
        for output_socket in self.core_component.output_sockets:  # type: Socket
            # Create empty out-sock registry entry.
            self.out_socket_registry[output_socket.name] = set()

            assert len(output_socket.op_records) > 0, "ERROR: There must at least be one op-record for out-Socket " \
                                                      "'{}'!".format(output_socket.name)

            # Loop through this Socket's set of possible ops.
            for op_rec in output_socket.op_records:
                # Get all the (core) in-Socket names (alphabetically sorted) that are required for this op.
                sockets = tuple(sorted(list(self.trace_back_sockets({op_rec})), key=lambda s: s.name))
                # If an in-Socket has more than one connected incoming Space:
                # Get the shape-combinations for these Sockets.
                # e.g. Sockets=["a", "b"] (and Space1 -> a, Space2 -> a, Space3 -> b)
                #   shape-combinations=[(Space1, Space3), (Space2, Space3)]
                shapes = [[i.get_shape(with_batch_rank=True) for i in sock.incoming_connections] for sock in sockets]
                shape_combinations = itertools.product(*shapes)
                for shape_combination in shape_combinations:
                    # Do everything by Socket-name (easier to debug).
                    in_socket_names = tuple([s.name for s in sockets])
                    # Update our call registry.
                    key = (output_socket.name, in_socket_names, shape_combination)
                    self.call_registry[key] = op_rec.op
                    # .. and the out-socket registry.
                    self.out_socket_registry[output_socket.name].update(set(in_socket_names))

    def sanity_check_build(self, component=None):
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

    def get_execution_inputs(self, output_socket_names, inputs=None):
        """
        Fetches graph api_methods for execution.

        Args:
            output_socket_names (Union[str,List[str]]): A name or a list of names of the out-Sockets to fetch from
            our core component.
            inputs (Optional[dict,data]): Dict specifying the provided api_methods for some in-Sockets.
                Depending on these given api_methods, the correct backend-ops can be selected within the given (out)-Sockets.
                Alternatively, can pass in data directly (not as a dict), but only if there is only one in-Socket in the
                Model or only one of the in-Sockets is needed for the given out-Sockets.

        Returns:
            tuple: fetch-dict, feed-dict with relevant args.
9       """
        output_socket_names = force_list(output_socket_names)

        # Sanity check out-Socket names.
        for out_sock_name in output_socket_names:
            if out_sock_name not in self.out_socket_registry:
                raise YARLError("ERROR: Out-Socket '{}' not found in Model! Make sure you are fetching by the \n"
                                "correct out-Socket name.".format(out_sock_name))

        only_input_socket_name = None  # the name of the only in-Socket possible here
        # Some input is given.
        if inputs is not None:
            # Get only in-Socket ..
            if len(self.core_component.input_sockets) == 1:
                only_input_socket_name = self.core_component.input_sockets[0].name
            # .. or only in-Socket for single(!), given out-Socket.
            elif len(output_socket_names) == 1 and \
                    len(self.out_socket_registry[output_socket_names[0]]) == 1:
                only_input_socket_name = next(iter(self.out_socket_registry[output_socket_names[0]]))

            # Check whether data is given directly.
            if not isinstance(inputs, dict):
                if only_input_socket_name is None:
                    raise YARLError("ERROR: Input data (`api_methods`) given directly (not as dict) AND more than one \n"
                                    "in-Socket in Model OR more than one in-Socket needed for given out-Sockets '{}'!".
                                    format(output_socket_names))
                inputs = {only_input_socket_name: inputs}
            # Is a dict: Check whether it's a in-Socket name dict (leave as is) or a
            # data dict (add in-Socket name as key).
            else:
                # We have more than one necessary in-Sockets (leave as is) OR
                # the only necessary in-Socket name is not key of the dict -> wrap it.
                if only_input_socket_name is not None and only_input_socket_name not in inputs:
                    inputs = {only_input_socket_name: inputs}

            # Try all possible input combinations to see whether we got an op for that.
            # Input Socket names will be sorted alphabetically and combined from short sequences up to longer ones.
            # Example: api_methods={A: ..., B: ... C: ...}
            #   input_combinations=[ABC, AB, AC, BC, A, B, C]

            # These combinations have been memoized for fast lookup.
            key = tuple(sorted(inputs.keys()))
            input_combinations = self.input_combinations.get(key)
            if not input_combinations:
                raise YARLError("ERROR: At least one of the given in-Socket names {} seems to be non-existent "
                                "in Model!".format(key))

        # No input given (maybe an out-Socket that doesn't require input).
        else:
            input_combinations = list(())

        # Go through each (core) out-Socket names and collect the correct ops to go into the fetch_list.
        fetch_list = list()
        feed_dict = dict()
        for out_socket_name in output_socket_names:
            # Updates with relevant ops
            fetch_list, feed_dict = self._get_execution_inputs_for_socket(
                out_socket_name, input_combinations, fetch_list, inputs, feed_dict)
        return fetch_list, feed_dict

    def _get_execution_inputs_for_socket(self, socket_name, input_combinations, fetch_list, input_dict, feed_dict):
        """
        Helper (to avoid nested for loop-break) for the loop in get_execution_inputs.

        Args:
            socket_name (str): The name of the (core) out-Socket to process.
            input_combinations (List[str]): The list of in-Socket (names) combinations starting with the combinations
                with the most Socket names, then going towards combinations with only one Socket name.
                Each combination in itself should already be sorted alphabetically on the in-Socket names.
            fetch_list (list): Appends to this list, which ops to actually fetch.
            input_dict (Optional[dict]): Dict specifying the provided api_methods for some (core) in-Sockets.
                Passed through directly from the call method.
            feed_dict (dict): The feed_dict we are trying to build. When done,
                needs to map input ops (not Socket names) to data.

        Returns:
            tuple: fetch_list, feed-dict with relevant args.
        """
        if len(input_combinations) > 0:
            # Check all (input+shape)-combinations and it we find one that matches what the user passed in as
            # `input_dict` -> Take that one and move on to the next Socket by returning.
            for input_combination in input_combinations:
                # Get all Space-combinations (in-op) for this input combination
                # (OBSOLETE: not possible anymore: in case an in-Socket has more than one connected incoming Spaces).
                ops = [self.in_socket_registry[c] for c in input_combination]
                # Get the shapes for this op_combination.
                shapes = tuple(get_shape(op) for op in ops)
                key = (socket_name, input_combination, shapes)
                # This is a good combination -> Use the looked up op, return to process next out-Socket.
                if key in self.call_registry:
                    fetch_list.append(self.call_registry[key])
                    # Add items to feed_dict.
                    for in_sock_name, in_op in zip(input_combination, ops):
                        value = input_dict[in_sock_name]
                        # Numpy'ize scalar values (tf doesn't sometimes like python primitives).
                        if isinstance(value, (float, int, bool)):
                            value = np.array(value)
                        feed_dict[in_op] = value
                    return fetch_list, feed_dict
        # No api_methods -> Try whether this output socket comes without any api_methods.
        else:
            key = (socket_name, (), ())
            if key in self.call_registry:
                fetch_list.append(self.call_registry[key])
                return fetch_list, feed_dict

        required_inputs = [k[1] for k in self.call_registry.keys() if k[0] == socket_name]
        raise YARLError("ERROR: No op found for out-Socket '{}' given the input-combinations: {}! "
                        "The following input-combinations are required for '{}':\n"
                        "{}".format(socket_name, input_combinations, socket_name, required_inputs))

    def trace_back_sockets(self, trace_set):
        """
        For a set of given ops, returns a list of all (core) in-Sockets that are required to calculate these ops.

        Args:
            trace_set (Set[Union[DataOpRecords,Socket]]): The set of DataOpRecord/Socket objects to trace-back till
                the beginning of the Graph. Socket entries mean we have already reached the beginning of the Graph and
                these will no further be traced back.

        Returns:
            Set[Socket]: in-Socket objects (from the core Component) that are required to calculate the DataOps
                in `trace_set`.
        """
        # Recursively lookup op in op_record_registry until we hit a Socket.
        new_trace_set = set()
        for op_rec_or_socket in trace_set:
            # We hit a Socket (we reached the beginning of the Graph). Stop tracing further back.
            if isinstance(op_rec_or_socket, Socket):
                if op_rec_or_socket.name not in self.in_socket_registry:
                    raise YARLError("ERROR: in-Socket '{}' could not be found in in_socket_registry of "
                                    "model!".format(op_rec_or_socket.name))
                new_trace_set.add(op_rec_or_socket)
            # A DataOpRecord: Sanity check that we already have this.
            elif op_rec_or_socket not in self.op_record_registry:
                # Could be a DataOpRecord of a SingleDataOp with constant_value set.
                if not isinstance(op_rec_or_socket.op, SingleDataOp) or op_rec_or_socket.op.constant_value is None:
                    raise YARLError("ERROR: DataOpRecord for op '{}' could not be found in op_record_registry of "
                                    "model!".format(op_rec_or_socket.op))
            else:
                new_trace_set.update(self.op_record_registry[op_rec_or_socket])
        if all([isinstance(i, Socket) for i in new_trace_set]):
            return new_trace_set
        else:
            return self.trace_back_sockets(new_trace_set)

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
