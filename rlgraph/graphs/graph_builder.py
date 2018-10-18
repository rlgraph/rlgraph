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

import logging
import re
import time

import inspect
from rlgraph import get_backend
from rlgraph.utils.rlgraph_errors import RLGraphError
from rlgraph.utils.specifiable import Specifiable

from rlgraph.components.component import Component
from rlgraph.spaces import Space, Dict
from rlgraph.spaces.space_utils import get_space_from_op, check_space_equivalence
from rlgraph.utils.input_parsing import parse_summary_spec
from rlgraph.utils.util import force_list, force_tuple, get_shape
from rlgraph.utils.ops import DataOpTuple, is_constant
from rlgraph.utils.op_records import FlattenedDataOp, DataOpRecord, DataOpRecordColumnIntoGraphFn, \
    DataOpRecordColumnIntoAPIMethod, DataOpRecordColumnFromGraphFn, get_call_param_name
from rlgraph.utils.component_printout import component_print_out

if get_backend() == "tf":
    import tensorflow as tf
    from rlgraph.utils.tf_util import pin_global_variables


class GraphBuilder(Specifiable):
    """
    The graph builder assembles the RLGraph meta-graph by tracing through
    components, sockets and connections and creating the underlying computation
    graph.
    """
    def __init__(self, name="model", action_space=None, summary_spec=None):
        """
        Args:
            name (str): The name of this GraphBuilder and of the meta-graph's root-component.
            action_space (Optional[Space]): The action Space information to be passed into calls to each Components'
                `when_input_complete` methods.
            summary_spec (Optional[dict]): A specification dict that defines, which summaries we would like to
                create in the graph and register with each Component.
        """
        super(GraphBuilder, self).__init__()

        # The name of this model. Our root-Component gets this name.
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

        # Build status/phase. Can take values
        # None: Build has not started yet.
        # "building": actual graph is being built from meta-graph.
        self.phase = None

        # Some build stats:
        # Number of meta op-records.
        self.num_meta_ops = 0
        # Number of actual backend ops.
        self.num_ops = 0
        # Number of trainable variables (optimizable weights).
        self.num_trainable_parameters = 0

        # Create an empty root-Component into which everything will be assembled by an Algo.
        self.root_component = None

        # Maps API method names to in- (placeholders) and out op columns (ops to pull).
        self.api = dict()

        self.op_records_to_process = set()
        self.op_records_to_process_later = set()

        # Dict of unprocessed (and complete) op-record columns by key=op-column ID.
        # Columns that have been forwarded will be erased again from this collection.
        # NOT NEEDED SO FAR.
        # self.unprocessed_in_op_columns = OrderedDict()
        # self.unprocessed_complete_in_op_columns = OrderedDict()

    def build_graph_with_options(self, meta_graph, input_spaces, available_devices,
                                 device_strategy="default", default_device=None,
                                 device_map=None, build_options=None):
        """
        Builds graph with the given options. See build doc for build details.

        Args:
            meta_graph (MetaGraph): MetaGraph to build to backend graph.
            input_spaces (dict): Input spaces to build for.
            available_devices (list): Devices which can be used to assign parts of the graph
                during graph assembly.
            device_strategy (Optional[str]): Device strategy.
            default_device (Optional[str]): Default device identifier.
            device_map (Optional[Dict]): Dict of Component names mapped to device names to place the Component's ops.
            build_options (Optional[Dict]): Dict of build options, e.g. default device handling for TF.
        """
        # No build options.
        if build_options is None:
            return self.build_graph(meta_graph, input_spaces, available_devices,
                                    device_strategy, default_device, device_map)
        else:
            if get_backend() == "tf":
                # Need to be fully specified to avoid errors, no defaults.
                default_device_context = build_options["build_device_context"]
                pin_global = build_options["pin_global_variable_device"]
                if pin_global is not None:
                    # Pin global variables for distributed TF.
                    with tf.device(default_device_context), \
                         pin_global_variables(pin_global):
                        return self.build_graph(meta_graph, input_spaces, available_devices,
                                                device_strategy, default_device, device_map)
                else:
                    with tf.device(default_device_context):
                        return self.build_graph(meta_graph, input_spaces, available_devices,
                                                device_strategy, default_device, device_map)
            else:
                raise RLGraphError("Build options are currently only available for TensorFlow.")

    def build_graph(self, meta_graph, input_spaces, available_devices,
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
            meta_graph (MetaGraph): MetaGraph to build to backend graph.
            input_spaces (dict): Input spaces to build for.
            available_devices (list): Devices which can be used to assign parts of the graph
                during graph assembly.
            device_strategy (Optional[str]): Device strategy.
            default_device (Optional[str]): Default device identifier.
            device_map (Optional[Dict]): Dict of Component names mapped to device names to place the Component's ops.
        """
        # Time the build procedure.
        time_start = time.perf_counter()
        assert meta_graph.build_status, "ERROR: Meta graph must be built to build backend graph."
        self.root_component = meta_graph.root_component
        self.graph_call_times = list()
        self.var_call_times = list()
        self.api = meta_graph.api
        self.num_meta_ops = meta_graph.num_ops

        # Set the build phase to `building`.
        self.phase = "building"

        # Set devices usable for this graph.
        self.available_devices = available_devices
        self.device_strategy = device_strategy
        self.default_device = default_device
        self.device_map = device_map or dict()

        # Create the first actual ops based on the input-spaces.
        # Some ops can only be created later when variable-based-Spaces are known (op_records_to_process_later).
        self.build_input_space_ops(input_spaces)

        # Collect all components and add those op-recs to the set that are constant.
        components = self.root_component.get_all_sub_components()
        for component in components:
            component.graph_builder = self  # point to us.
            self.op_records_to_process.update(component.constant_op_records)
            # Check whether the Component is input-complete (and build already if it is).
            self.build_component_when_input_complete(component)

        op_records_list = sorted(self.op_records_to_process, key=lambda rec: rec.id)

        # Re-iterate until our bag of op-recs to process is empty.
        iterations = self._build(op_records_list)
        time_build = time.perf_counter() - time_start
        self.logger.info("Computation-Graph build completed in {} s ({} iterations).".format(time_build, iterations))

        # Get some stats on the graph and report.
        self.num_ops = self.count_ops()
        self.logger.info("Actual graph ops generated: {}".format(self.num_ops))

        self.num_trainable_parameters = self.count_trainable_parameters()
        self.logger.info("Number of trainable parameters: {}".format(self.num_trainable_parameters))
        # The build here is the actual build overhead, so build time minus the tensorflow calls and variable
        # creations which would have to happen either way.
        build_overhead = time_build - sum(self.graph_call_times) - sum(self.var_call_times)

        return dict(
            build_overhead=build_overhead,
            total_build_time=time_build,
            op_creation=sum(self.graph_call_times),
            var_creation=sum(self.var_call_times)
        )

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
        for api_method_name, (in_op_records, _) in sorted(self.api.items()):
            api_method_rec = self.root_component.api_methods[api_method_name]
            spaces = list()
            for param_name in api_method_rec.input_names:
                if self.root_component.api_method_inputs[param_name] == "flex":
                    if input_spaces is not None and param_name in input_spaces:
                        spaces.append(input_spaces[param_name])
                        self.root_component.api_method_inputs[param_name] = input_spaces[param_name]
                elif isinstance(self.root_component.api_method_inputs[param_name], Space):
                    if input_spaces is not None and param_name in input_spaces:
                        spaces.append(self.root_component.api_method_inputs[param_name])
                else:
                    if self.root_component.api_method_inputs[param_name] == "*flex":
                        if param_name in input_spaces:
                            spaces.extend(force_list(input_spaces[param_name]))
                    elif self.root_component.api_method_inputs[param_name] == "**flex":
                        if param_name in input_spaces:
                            spaces.extend([
                                input_spaces[param_name][k] for k in sorted(input_spaces[param_name].keys())
                            ])
                    else:
                        assert param_name in input_spaces
                        spaces.append(input_spaces[param_name])
            assert len(spaces) == len(in_op_records)

            # Create the placeholder and store it in the given DataOpRecords.
            for i, space in enumerate(spaces):
                # Space is dependent on the variables of some sub-component (wait with the construction of the
                # placeholder until after the build).
                if isinstance(space, str) and re.match(r'^variables:', space):
                    in_op_records[i].space = space
                    self.op_records_to_process_later.add(in_op_records[i])
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
                self.op_records_to_process.add(in_op_records[i])

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
        device = self.get_device(component)  #, variables=True)
        placeholder = None
        if get_backend() == "tf":
            with tf.device(device):
                placeholder = space.get_variable(name=name, is_input_feed=True)
        elif get_backend() == "pytorch":
                # Batch rank 1 because PyTorch does not allow None shapes.
                placeholder = space.get_variable(name=name, add_batch_rank=1,
                                                 is_input_feed=True, is_python=True)
        return placeholder

    def build_component_when_input_complete(self, component):
        # Not input complete yet -> Check now.
        if component.input_complete is False:
            component.check_input_completeness()
            # Call `when_input_complete` once on that Component.
            if component.input_complete is True:
                self.logger.debug("Component {} is input-complete; Spaces per API-method input parameter are: {}".
                                  format(component.name, component.api_method_inputs))
                device = self.get_device(component, variables=True)
                # This builds variables which would have to be done either way:
                call_time = time.perf_counter()
                component.when_input_complete(input_spaces=None, action_space=self.action_space, device=device)
                self.var_call_times.append(time.perf_counter() - call_time)
                # Call all no-input graph_fns of the new Component.
                for no_in_col in component.no_input_graph_fn_columns:
                    # Do not call _variables (only later, when Component is also variable-complete).
                    if no_in_col.graph_fn.__name__ != "_graph_fn__variables":
                        self.run_through_graph_fn_with_device_and_scope(no_in_col)
                        # Keep working with the generated output ops.
                        self.op_records_to_process.update(no_in_col.out_graph_fn_column.op_records)

                # Now that the component is input-complete, the parent may have become variable-complete.
                if component.parent_component is not None:
                    self.build_component_when_input_complete(component.parent_component)

        # Check variable-completeness and actually call the _variable graph_fn if the component just became
        # "variable-complete".
        if component.input_complete is True and component.variable_complete is False and \
                component.check_variable_completeness():
            # The graph_fn _variables has some in-op-columns that need to be run through the function.
            if "_graph_fn__variables" in component.graph_fns:
                graph_fn_rec = component.graph_fns["_graph_fn__variables"]
                # TODO: Think about only running through no-input-graph-fn once, no matter how many in-op-columns it has.
                # TODO: Then link the first in-op-column (empty) to all out-op-columns.
                for i, in_op_col in enumerate(graph_fn_rec.in_op_columns):
                    if in_op_col.already_sent is False:
                        self.run_through_graph_fn_with_device_and_scope(in_op_col)
                        # Keep working with the generated output ops.
                        self.op_records_to_process.update(graph_fn_rec.out_op_columns[i].op_records)

    def run_through_graph_fn_with_device_and_scope(self, op_rec_column, create_new_out_column=False):
        """
        Runs through a graph_fn with the given ops and thereby assigns a device (Component's device or GraphBuilder's
        default) to the ops generated by a graph_fn.

        Args:
            op_rec_column (DataOpRecordColumnIntoGraphFn): The column of DataOpRecords to be fed through the
                graph_fn.
            create_new_out_column (bool): Whether to produce the out op-record column (or use the one already in
                the meta-graph). If True and the `op_rec_column` already links to an out op-rec column, raises
                an error.
                Default: False.
        """
        # Get the device for the ops generated in the graph_fn (None for custom device-definitions within the graph_fn).
        device = self.get_device(op_rec_column.component, variables=False)

        if get_backend() == "tf":
            # TODO: Write custom scope generator for devices (in case None, etc..).
            if device is not None:
                # Assign proper device to all ops created in this context manager.
                with tf.device(device):
                    # Name ops correctly according to our Component hierarchy.
                    with tf.name_scope(op_rec_column.component.global_scope +
                                       ('/' if op_rec_column.component.global_scope else "")):
                        self.logger.info(
                            "Assigning device '{}' to graph_fn '{}' (scope '{}').".
                            format(device, op_rec_column.graph_fn.__name__, op_rec_column.component.global_scope)
                        )
                        out_op_rec_column = self.run_through_graph_fn(
                            op_rec_column, create_new_out_column=create_new_out_column
                        )
                        op_rec_column.out_graph_fn_column = out_op_rec_column
            else:
                # Name ops correctly according to our Component hierarchy.
                with tf.name_scope(op_rec_column.component.global_scope +
                                   ('/' if op_rec_column.component.global_scope else "")):
                    out_op_rec_column = self.run_through_graph_fn(
                        op_rec_column, create_new_out_column=create_new_out_column
                    )
                    op_rec_column.out_graph_fn_column = out_op_rec_column

            # Store assigned names for debugging.
            if device is not None:
                if device not in self.device_component_assignments:
                    self.device_component_assignments[device] = [str(op_rec_column.graph_fn.__name__)]
                else:
                    self.device_component_assignments[device].append(str(op_rec_column.graph_fn.__name__))

        elif get_backend() == "pytorch":
            # No device handling via decorators.
            out_op_rec_column = self.run_through_graph_fn(
                op_rec_column, create_new_out_column=create_new_out_column
            )
            op_rec_column.out_graph_fn_column = out_op_rec_column

        # Tag column as already sent through graph_fn.
        op_rec_column.already_sent = True  # TODO: assert is False before this?
        return op_rec_column.out_graph_fn_column

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
            device = device.get("variables", None) if variables is True else device.get("ops", None)

        # If device is local, but not available, use the default device (or None).
        # TODO rethink handling device maps
        # if device is not None and not re.match(r'^/job:', device) and device not in self.available_devices:
        #     device = self.device_map.get(component.name, self.default_device)
        # Device is specific to whether we are creating variables or ops.
        if isinstance(device, dict):
            device = device.get("variables", None) if variables is True else device.get("ops", None)

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
            api_input_records = self.root_component.api_methods[api_method].in_op_columns[0].op_records
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
                            self.logger.debug("Subgraph component: current op record component: {}, next: {}".format(
                                op_rec.column.component, next_op_rec.column.component
                            ))
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

    def run_through_graph_fn(self, op_rec_column, create_new_out_column=False):
        """
        Pushes all ops in the column through the respective graph_fn (graph_fn-spec and call-options are part of
        the column).
        Call options include flattening ops, flattening+splitting ops and (when splitting) adding the auto-generated
        flat key as first parameter to the different (split) calls of graph_fn.
        After the call, the already existing output column is populated with the actual results.

        Args:
            op_rec_column (DataOpRecordColumnIntoGraphFn): The column of DataOpRecords to be fed through the
                graph_fn.
            create_new_out_column (bool): Whether to produce the out op-record column (or use the one already in
                the meta-graph). If True and the `op_rec_column` already links to an out op-rec column, raises
                an error.
                Default: False.

        Returns:
            DataOpRecordColumnFromGraphFn: The op-record column coming out of the graph_fn. This column may have
                already existed in the meta-graph before the graph_fn call or may have been generated during this
                call (if `create_new_out_column` is True).
        """
        args = [r.op for r in op_rec_column.op_records if r.kwarg is None]
        kwargs = {r.kwarg: r.op for r in op_rec_column.op_records if r.kwarg is not None}
        assert all(op is not None for op in args)  # just make sure

        call_time = None

        # Build the ops from this input-combination.
        # Flatten input items.
        if op_rec_column.flatten_ops is not False:
            flattened_args, flattened_kwargs = op_rec_column.flatten_input_ops(*args, **kwargs)
            # Split into SingleDataOps?
            if op_rec_column.split_ops:
                split_args_and_kwargs = op_rec_column.split_flattened_input_ops(*flattened_args, **flattened_kwargs)
                # There is some splitting to do. Call graph_fn many times (one for each split).
                if isinstance(split_args_and_kwargs, FlattenedDataOp):
                    ops = dict()
                    num_return_values = -1
                    for key, params in split_args_and_kwargs.items():
                        params_args = [p for p in params if not isinstance(p, tuple)]
                        params_kwargs = {p[0]: p[1] for p in params if isinstance(p, tuple)}
                        if create_new_out_column is False:
                            call_time = time.perf_counter()
                        ops[key] = force_tuple(op_rec_column.graph_fn(op_rec_column.component,
                                                                      *params_args, **params_kwargs))
                        if create_new_out_column is False:
                            self.graph_call_times.append(time.perf_counter() - call_time)
                        if num_return_values >= 0 and num_return_values != len(ops[key]):
                            raise RLGraphError(
                                "Different split-runs through {} do not return the same number of values!".
                                format(op_rec_column.graph_fn.__name__)
                            )
                        num_return_values = len(ops[key])
                    # Un-split the results dict into a tuple of `num_return_values` slots.
                    un_split_ops = list()
                    for i in range(num_return_values):
                        dict_with_singles = FlattenedDataOp()
                        for key in split_args_and_kwargs.keys():
                            dict_with_singles[key] = ops[key][i]
                        un_split_ops.append(dict_with_singles)
                    ops = tuple(un_split_ops)

                # No splitting to do: Pass everything as-is.
                else:
                    split_args, split_kwargs = split_args_and_kwargs[0], split_args_and_kwargs[1]
                    if create_new_out_column is False:
                        call_time = time.perf_counter()
                    ops = op_rec_column.graph_fn(op_rec_column.component, *split_args, **split_kwargs)
                    if create_new_out_column is False:
                        self.graph_call_times.append(time.perf_counter() - call_time)
            else:
                if create_new_out_column is False:
                    call_time = time.perf_counter()
                ops = op_rec_column.graph_fn(op_rec_column.component, *flattened_args, **flattened_kwargs)
                if create_new_out_column is False:
                    self.graph_call_times.append(time.perf_counter() - call_time)
        # Just pass in everything as-is.
        else:
            if create_new_out_column is False:
                call_time = time.perf_counter()
            ops = op_rec_column.graph_fn(op_rec_column.component, *args, **kwargs)
            if create_new_out_column is False:
                self.graph_call_times.append(time.perf_counter() - call_time)

        # Make sure everything coming from a computation is always a tuple (for out-Socket indexing).
        ops = force_tuple(ops)

        # Always un-flatten all return values. Otherwise, we would allow Dict Spaces
        # with '/' keys in them, which is not allowed.
        ops = op_rec_column.unflatten_output_ops(*ops)

        # Should we create a new out op-rec column?
        if create_new_out_column is True:
            # Assert that we don't have an out column already (wouldn't make sense).
            assert op_rec_column.out_graph_fn_column is None
            out_graph_fn_column = DataOpRecordColumnFromGraphFn(
                len(ops), component=op_rec_column.component, graph_fn_name=op_rec_column.graph_fn.__name__,
                in_graph_fn_column=op_rec_column
            )
        else:
            out_graph_fn_column = op_rec_column.out_graph_fn_column
            # Make sure the number of returned ops matches the number of op-records in the next column.
            assert len(ops) == len(out_graph_fn_column.op_records), \
                "ERROR: Number of returned values of graph_fn '{}/{}' ({}) does not match the number of op-records " \
                "({}) reserved for the return values of the method!".format(
                    op_rec_column.component.name, op_rec_column.graph_fn.__name__, len(ops),
                    len(out_graph_fn_column.op_records)
                )

        # Determine the Spaces for each out op and then move it into the respective op and Space slot of the
        # out_graph_fn_column.
        for i, op in enumerate(ops):
            space = get_space_from_op(op)
            # Make sure the receiving op-record is still empty.
            assert out_graph_fn_column.op_records[i].op is None
            out_graph_fn_column.op_records[i].op = op
            out_graph_fn_column.op_records[i].space = space

        return out_graph_fn_column

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
                num_trainable_parameters += get_shape(variable, flat=True)

        return num_trainable_parameters

    @staticmethod
    def count_ops():
        """
        Counts the number of all backend-specific ops present in the graph. This includes variables and placeholders.

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
        component = component or self.root_component

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

    def get_execution_inputs(self, *api_method_calls):
        """
        Creates a fetch-dict and a feed-dict for a graph session call.

        Args:
            api_method_calls (dict): See `rlgraph.graphs.graph_executor` for details.

        Returns:
            Tuple[list,dict]: Fetch-list, feed-dict with relevant args.
        """
        fetch_dict = {}
        feed_dict = {}

        for api_method_call in api_method_calls:
            if api_method_call is None:
                continue

            params = []
            return_ops = None
            if isinstance(api_method_call, (list, tuple)):
                params = force_list(api_method_call[1])
                return_ops = force_list(api_method_call[2]) if len(api_method_call) > 2 and \
                                                               api_method_call[2] is not None else None
                api_method_call = api_method_call[0]

            if api_method_call not in self.api:
                raise RLGraphError("No API-method with name '{}' found!".format(api_method_call))

            # API returns a dict.
            if len(self.api[api_method_call][1]) > 0 and self.api[api_method_call][1][0].kwarg is not None:
                fetch_dict[api_method_call] = {op_rec.kwarg: op_rec.op for op_rec in self.api[api_method_call][1] if
                                               return_ops is None or op_rec.kwarg in return_ops}
            # API returns a tuple.
            else:
                fetch_dict[api_method_call] = [op_rec.op for i, op_rec in enumerate(self.api[api_method_call][1]) if
                                               return_ops is None or i in return_ops]

            for i, param in enumerate(params):
                # TODO: What if len(params) < len(self.api[api_method][0])? Need to handle default API-method params also for the root-component (this one).
                if len(self.api[api_method_call][0]) <= i:
                    raise RLGraphError("API-method with name '{}' only has {} input parameters! You passed in "
                                       "{}.".format(api_method_call, len(self.api[api_method_call][0]), len(params)))

                placeholder = self.api[api_method_call][0][i].op  # 0=input op-recs; i=ith input op-rec
                if isinstance(placeholder, DataOpTuple):
                    for ph, p in zip(placeholder, param):
                        feed_dict[ph] = p
                # Special case: Get the default argument for this arg.
                # TODO: Support API-method's kwargs here as well (mostly useful for test.test).
                #elif param is None:
                #    feed_dict[placeholder] = self.root_component.api_methods[api_method].default_values[i]
                else:
                    feed_dict[placeholder] = param

        return fetch_dict, feed_dict

    def execute_define_by_run_op(self, api_method, params=None):
        """
        Executes an API method by simply calling the respective function
        directly with its parameters to trigger an eager call-chain through the graph.

        Args:
            api_method (str): Name of api-method.
            params (Optional[list]): Optional arguments.

        Returns:
            any: Results of executing this api-method.
        """
        # Reset call profiler.
        Component.reset_profile()
        if api_method not in self.api:
            raise RLGraphError("No API-method with name '{}' found!".format(api_method))

        if params is not None:
            if api_method in self.root_component.synthetic_methods:
                return self.root_component.api_fn_by_name[api_method](self.root_component, *params)
            else:
                return self.root_component.api_fn_by_name[api_method](*params)
        else:
            if api_method in self.root_component.synthetic_methods:
                return self.root_component.api_fn_by_name[api_method](self.root_component)
            else:
                return self.root_component.api_fn_by_name[api_method]()

    def build_define_by_run_graph(self, meta_graph, input_spaces, available_devices,
                                  device_strategy="default", default_device=None, device_map=None):
        """
        Builds a graph for eager execution. This primarily consists of creating variables through
        the component hierarchy by pushing the input spaces  through the graph.

          Args:
            meta_graph (MetaGraph): MetaGraph to build to backend graph.
            input_spaces (dict): Input spaces to build for.
            available_devices (list): Devices which can be used to assign parts of the graph
                during the graph build.
            device_strategy (Optional[str]): Device strategy.
            default_device (Optional[str]): Default device identifier.
            device_map (Optional[Dict]): Dict of Component names mapped to device names to place the Component's ops.
        """
        # Time the build procedure.
        time_start = time.perf_counter()
        assert meta_graph.build_status, "ERROR: Meta graph must be built to build backend graph."
        self.root_component = meta_graph.root_component
        self.graph_call_times = []
        self.var_call_times = []
        self.api = meta_graph.api
        self.num_meta_ops = meta_graph.num_ops

        # Set devices usable for this graph.
        self.available_devices = available_devices
        self.device_strategy = device_strategy
        self.default_device = default_device
        self.device_map = device_map or {}

        # TODO device strategy in pytorch?
        # Build full registry of callable methods on root component.
        for member in inspect.getmembers(self.root_component):
            name, method = (member[0], member[1])

            # N.b. this means _graph_fns are not directly callable here, just api functions.
            if name not in self.root_component.api_fn_by_name and name in self.api:
                self.root_component.api_fn_by_name[name] = method

        # Create the first actual ops based on the input-spaces.
        # Some ops can only be created later when variable-based-Spaces are known (op_records_to_process_later).
        self.build_input_space_ops(input_spaces)

        # Collect all components and add those op-recs to the set that are constant.
        components = self.root_component.get_all_sub_components()
        for component in components:
            component.graph_builder = self  # point to us.
            self.op_records_to_process.update(component.constant_op_records)
            # Check whether the Component is input-complete (and build already if it is).
            self.build_component_when_input_complete(component)

        op_records_list = sorted(self.op_records_to_process, key=lambda rec: rec.id)
        iterations = self._build(op_records_list)

        # Set execution mode in components to change `call` behaviour to direct function evaluation.
        self.root_component.propagate_sub_component_properties(properties=dict(execution_mode="define_by_run"))

        # Call post build logic.
        self.root_component._post_build(self.root_component)

        time_build = time.perf_counter() - time_start
        self.logger.info("Define-by-run computation-graph build completed in {} s ({} iterations).".
                         format(time_build, iterations))
        build_overhead = time_build - sum(self.graph_call_times) - sum(self.var_call_times)
        return dict(
            build_overhead=build_overhead,
            total_build_time=time_build,
            op_creation=sum(self.graph_call_times),
            var_creation=sum(self.var_call_times)
        )

    def _build(self, op_records_list):
        """
        Private implementation of the main build loop. For docs, see the respective build
        methods.
        """
        loop_counter = 0
        while len(op_records_list) > 0:
            # Collect op-recs to process in the next iteration.
            self.op_records_to_process = set()

            # Set of Components that have been tried last to get input-complete. If build gets stuck, it'll be because
            # of the Components in this set.
            non_complete_components = set()
            for op_rec in op_records_list:  # type: DataOpRecord
                # There are next records:
                if len(op_rec.next) > 0:
                    # Push actual op and Space forward one op-rec at a time.
                    for next_op_rec in sorted(op_rec.next, key=lambda rec: rec.id):  # type: DataOpRecord
                        # Assert that next-record's `previous` field points back to op_rec.
                        assert next_op_rec.previous is op_rec, \
                            "ERROR: Op-rec {} in meta-graph has {} as next, but {}'s previous field points to {}!". \
                            format(op_rec, next_op_rec, next_op_rec, next_op_rec.previous)
                        # If not last op in this API-method -> continue.
                        if next_op_rec.is_terminal_op is False:
                            assert next_op_rec.op is None or is_constant(next_op_rec.op)
                            self.op_records_to_process.add(next_op_rec)
                        # Push op and Space into next op-record.
                        next_op_rec.op = op_rec.op
                        next_op_rec.space = op_rec.space

                        # Also push Space into possible API-method record if slot's Space is still None.
                        if isinstance(op_rec.column, DataOpRecordColumnIntoAPIMethod):
                            param_name = get_call_param_name(op_rec)
                            component = op_rec.column.api_method_rec.component

                            # Place Space for this input-param name (valid for all input params of same name even of
                            # different API-method of the same Component).
                            if component.api_method_inputs[param_name] is None or \
                                    component.api_method_inputs[param_name] == "flex":
                                component.api_method_inputs[param_name] = next_op_rec.space
                            # Sanity check, whether Spaces are equivalent.
                            else:
                                generic_space = check_space_equivalence(
                                    component.api_method_inputs[param_name], next_op_rec.space
                                )
                                # Spaces are not equivalent.
                                if generic_space is False:
                                    raise RLGraphError(
                                        "ERROR: op-rec '{}' has Space '{}', but input-param '{}' already has Space "
                                        "'{}'!".format(next_op_rec, next_op_rec.space, param_name,
                                                       component.api_method_inputs[param_name])
                                    )
                                # Overwrite both entries with the more generic Space.
                                else:
                                    next_op_rec.space = component.api_method_inputs[param_name] = \
                                        generic_space

                        # Did we enter a new Component? If yes, check input-completeness and
                        # - If op_rec.column is None -> We are at the very beginning of the graph (op_rec.op is a
                        # placeholder).
                        next_component = next_op_rec.column.component
                        if op_rec.column is None or op_rec.column.component is not next_component:
                            self.build_component_when_input_complete(next_component)
                            if next_component.input_complete is False:
                                non_complete_components.add(next_component.global_scope)

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
                            self.op_records_to_process.update(op_rec.column.out_graph_fn_column.op_records)
                        # Component not input-complete. Keep coming back with this op.
                        else:
                            self.build_component_when_input_complete(op_rec.column.component)
                            self.op_records_to_process.add(op_rec)
                            if op_rec.column.component.input_complete is False:
                                non_complete_components.add(op_rec.column.component.global_scope)
                    # - Op column is not complete yet: Discard this one (as others will keep coming in anyway).
                    # - Op column has already been sent (sibling ops may have arrived in same iteration).
                # - Op belongs to a column coming from a graph_fn or an API-method, but the op is no longer used.
                # -> Ignore Op.

            # Sanity check, whether we are stuck.
            new_op_records_list = sorted(self.op_records_to_process, key=lambda rec: rec.id)
            if op_records_list == new_op_records_list:
                # Ok for some
                if loop_counter > 10000:
                    #print()
                    raise RLGraphError(
                        "Build procedure is deadlocked. Most likely, you are having a circularly dependent Component "
                        "in your meta-graph. The following Components are still input-incomplete:\n{}".
                        format(non_complete_components)
                    )

            op_records_list = new_op_records_list

            # If we are done with the build, check for API-methods' ops that are dependent on variables
            # generated during the build and build these now.
            # TODO is this loop necessary for define by run?
            if get_backend() == "tf":
                if len(op_records_list) == 0 and self.op_records_to_process_later is not None and \
                        len(self.op_records_to_process_later) > 0:
                    op_records_list = list(self.op_records_to_process_later)
                    # Invalidate later-set.
                    self.op_records_to_process_later = None  # type: set

                    # Loop through the op_records list and sanity check for "variables"-dependent Spaces, then get these
                    # Spaces, create the placeholders and keep building.
                    for op_rec in op_records_list:
                        space_desc = op_rec.space  # type: str
                        mo = re.search(r'^variables:(.+)', space_desc)
                        assert mo
                        component_path = mo.group(1).split("/")
                        component = self.root_component
                        for level in component_path:
                            assert level in component.sub_components, \
                                "ERROR: `component_path` ('{}') contains non-existent Components!".format(component_path)
                            component = component.sub_components[level]
                        var_spaces = {key: get_space_from_op(value) for key, value in sorted(
                            component.get_variables(custom_scope_separator="-").items()
                        )}
                        var_space = Dict(var_spaces)
                        op_rec.space = var_space
                        op_rec.op = self.get_placeholder("api-", space=var_space, component=self.root_component)

            loop_counter += 1
        return loop_counter
