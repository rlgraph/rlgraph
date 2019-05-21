# Copyright 2018/2019 The RLgraph authors. All Rights Reserved.
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

import inspect
import logging
import re
import time
from collections import OrderedDict

from rlgraph import get_backend
from rlgraph.components.component import Component
from rlgraph.spaces import Space, Dict
from rlgraph.spaces.space_utils import get_space_from_op, check_space_equivalence
from rlgraph.utils.define_by_run_ops import define_by_run_flatten, define_by_run_split_args, define_by_run_unflatten, \
    define_by_run_unpack
from rlgraph.utils.input_parsing import parse_summary_spec
from rlgraph.utils.op_records import FlattenedDataOp, DataOpRecord, DataOpRecordColumnIntoGraphFn, \
    DataOpRecordColumnIntoAPIMethod, DataOpRecordColumnFromGraphFn, DataOpRecordColumnFromAPIMethod, get_call_param_name
from rlgraph.utils.ops import is_constant, ContainerDataOp, DataOpDict, flatten_op, unflatten_op, TraceContext
from rlgraph.utils.rlgraph_errors import RLGraphError, RLGraphBuildError
from rlgraph.utils.specifiable import Specifiable
from rlgraph.utils.util import force_list, force_tuple, get_shape

if get_backend() == "tf":
    import tensorflow as tf
    from rlgraph.utils.tf_util import pin_global_variables
elif get_backend() == "pytorch":
    import torch


class GraphBuilder(Specifiable):
    """
    The graph builder assembles the RLGraph meta-graph by tracing through
    components, sockets and connections and creating the underlying computation
    graph.
    """
    def __init__(self, name="model", action_space=None, summary_spec=None, max_build_iterations=3000):
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
        self.action_space = None
        if action_space is not None:
            self.action_space = Space.from_spec(action_space)
        self.summary_spec = parse_summary_spec(summary_spec)
        self.max_build_iterations = max_build_iterations
        # Set of components that have been analyzed on why they remain input-incomplete.
        self.investigated_input_incomplete_components = set()

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
        self.graph_call_times = []
        self.var_call_times = []

        # Create an empty root-Component into which everything will be assembled by an Algo.
        self.root_component = None

        # Maps API method names to in- (placeholders) and out op columns (ops to pull).
        self.api = {}

        self.op_records_to_process = set()
        self.op_recs_depending_on_variables = set()

        # A register for all created placeholders by name.
        self.placeholders = {}

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
        # No device context options.
        if build_options is None or "build_device_context" not in build_options:
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
        self.graph_call_times = []
        self.var_call_times = []
        self.api = meta_graph.api
        self.num_meta_ops = meta_graph.num_ops

        # Set the build phase to `building`.
        self.phase = "building"

        # Set devices usable for this graph.
        self.available_devices = available_devices
        self.device_strategy = device_strategy
        self.default_device = default_device
        self.device_map = device_map or {}

        # Create the first actual ops based on the input-spaces.
        # Some ops can only be created later when variable-based-Spaces are known (op_recs_depending_on_variables).
        self.build_input_space_ops(input_spaces)

        # Collect all components and add those op-recs to the set that are constant.
        components = self.root_component.get_all_sub_components()
        # Point to this GraphBuilder object.
        # Add those op-recs to the set that are constant.
        for component in components:
            component.graph_builder = self
        # Try to build if already input complete.
        for component in components:
            self.op_records_to_process.update(component.constant_op_records)
            self.build_component_when_input_complete(component)

        op_records_list = self._sort_op_recs(self.op_records_to_process)

        # Re-iterate until our bag of op-recs to process is empty.
        iterations = self._build(op_records_list)
        time_build = time.perf_counter() - time_start
        self.logger.info("Computation-Graph build completed in {} s ({} iterations).".format(time_build, iterations))

        # Get some stats on the graph and report.
        self.num_ops = self.count_ops()
        self.logger.info("Actual graph ops generated: {}".format(self.num_ops))

        self.num_trainable_parameters = self.count_trainable_parameters()
        self.logger.info("Number of trainable parameters: {}".format(self.num_trainable_parameters))

        # Sanity check the build.
        self.sanity_check_build()

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
        if input_spaces is None:
            input_spaces = {}

        for api_method_name, (in_op_records, _) in sorted(self.api.items()):
            api_method_rec = self.root_component.api_methods[api_method_name]
            spaces = []
            for param_name in api_method_rec.input_names:
                if self.root_component.api_method_inputs[param_name] == "flex":
                    if param_name in input_spaces:
                        spaces.append((param_name, input_spaces[param_name]))
                        self.root_component.api_method_inputs[param_name] = Space.from_spec(input_spaces[param_name])
                elif isinstance(self.root_component.api_method_inputs[param_name], Space):
                    if param_name in input_spaces:
                        spaces.append((param_name, self.root_component.api_method_inputs[param_name]))
                else:
                    if self.root_component.api_method_inputs[param_name] == "*flex":
                        if param_name in input_spaces:
                            for i, s in enumerate(force_list(input_spaces[param_name])):
                                spaces.append((param_name+"-"+str(i), s))
                                self.root_component.api_method_inputs[param_name+"["+str(i)+"]"] = Space.from_spec(s)
                    elif self.root_component.api_method_inputs[param_name] == "**flex":
                        if param_name in input_spaces:
                            for k in sorted(input_spaces[param_name].keys()):
                                spaces.append((param_name+"-"+k, input_spaces[param_name][k]))
                                self.root_component.api_method_inputs[param_name+"["+k+"]"] = \
                                    Space.from_spec(input_spaces[param_name][k])
                    else:
                        assert param_name in input_spaces
                        spaces.append((param_name, input_spaces[param_name]))
                        self.root_component.api_method_inputs[param_name] = Space.from_spec(input_spaces[param_name])
            assert len(spaces) == len(in_op_records)

            # Create the placeholders and store them in the given DataOpRecords.
            for i, (name, space) in enumerate(spaces):
                # Space is dependent on the variables of some sub-component (wait with the construction of the
                # placeholder until the component is variable-complete).
                if isinstance(space, str) and re.match(r'^variables:', space):
                    in_op_records[i].space = space
                    self.op_recs_depending_on_variables.add(in_op_records[i])
                    continue

                # Construct Space from a spec.
                elif not isinstance(space, Space):
                    space = Space.from_spec(space)

                in_op_records[i].space = space

                in_op_records[i].op = self.get_placeholder(
                    name=name, space=space,
                    component=next(iter(in_op_records[0].next)).column.component
                )
                self.op_records_to_process.add(in_op_records[i])

    def get_placeholder(self, name, space, component):
        """
        Generates one or more placeholders given a name, space and a component (for device inference).

        Args:
            name (str): The name of the placeholder to create.
            space (spec(Space)): The Space object to generate the placeholder for.

            component (Component): The Component into which the placeholder will go (needed  for automatic device
                inference).

        Returns:
            DataOp: The generated placeholder(s) as a DataOp (e.g. DataOpTuple, SingleDataOp, etc..).
        """
        if name in self.placeholders:
            return self.placeholders[name]

        device = self.get_device(component)  #, variables=True)
        placeholder = None
        space = Space.from_spec(space)
        if get_backend() == "tf":
            with tf.device(device):
                placeholder = space.get_variable(name=name, is_input_feed=True)
        elif get_backend() == "pytorch":
                # Batch rank 1 because PyTorch does not allow None shapes.
                placeholder = space.get_variable(name=name, add_batch_rank=1,
                                                 is_input_feed=True, is_python=True)
        self.placeholders[name] = placeholder
        return placeholder

    def build_component_when_input_complete(self, component, check_sub_components=True):
        graph_fn_requiring_var_completeness = [gf.name for gf in component.graph_fns.values() if
                                               gf.requires_variable_completeness is True]
        # Not input complete yet -> Check now.
        if component.input_complete is False or component.built is False:
            component.check_input_completeness()
            # Call `when_input_complete` once on that Component.
            if component.input_complete is True:
                self.logger.debug("Component {} is input-complete; Spaces per API-method input parameter are: {}".
                                  format(component.name, component.api_method_inputs))
                device = self.get_device(component, variables=True)
                # This builds variables which would have to be done either way:
                call_time = time.perf_counter()
                component.when_input_complete(
                    input_spaces=None, action_space=self.action_space, device=device,
                    summary_regexp=self.summary_spec["summary_regexp"]
                )
                self.var_call_times.append(time.perf_counter() - call_time)
                # Call all no-input graph_fns of the new Component.
                for no_in_col in component.no_input_graph_fn_columns:
                    # Do not call _variables (only later, when Component is also variable-complete).
                    if no_in_col.graph_fn.__name__ not in graph_fn_requiring_var_completeness:
                        self.run_through_graph_fn_with_device_and_scope(no_in_col)
                        # Keep working with the generated output ops.
                        self.op_records_to_process.update(no_in_col.out_graph_fn_column.op_records)

        if component.input_complete is True and component.check_variable_completeness():
            # The graph_fn _variables has some in-op-columns that need to be run through the function.
            for graph_fn_name in graph_fn_requiring_var_completeness:
                graph_fn_rec = component.graph_fns[graph_fn_name]
                # TODO: Think about only running through no-input-graph-fn once, no matter how many in-op-columns it has.
                # TODO: Then link the first in-op-column (empty) to all out-op-columns.
                for i, in_op_col in enumerate(graph_fn_rec.in_op_columns):
                    if in_op_col.already_sent is False and in_op_col.is_complete():
                        self.run_through_graph_fn_with_device_and_scope(in_op_col)
                        # If graph_fn_rec doesn't know about the out-op-col yet, add it.
                        if len(graph_fn_rec.out_op_columns) <= i:
                            assert len(graph_fn_rec.out_op_columns) == i  # make sure, it's really just one col missing
                            graph_fn_rec.out_op_columns.append(in_op_col.out_graph_fn_column)
                        self.op_records_to_process.update(graph_fn_rec.out_op_columns[i].op_records)

            if check_sub_components is True:
                # Check variable-completeness and actually call the _variable graph_fn if not already done so.
                # Collect sub-components and build them as well if they just became variable-complete.
                sub_components = component.sub_components.values()
                sub_components_not_var_complete = set()
                for sub_component in sub_components:
                    if sub_component.variable_complete is False:
                        sub_components_not_var_complete.add(sub_component)

                for sub_component in sub_components_not_var_complete:
                    self.build_component_when_input_complete(sub_component)

            # Now that the component is variable-complete, the parent may have become variable-complete as well.
            if component.parent_component is not None and component.parent_component.variable_complete is False:
                self.build_component_when_input_complete(component.parent_component, check_sub_components=False)

    def run_through_graph_fn_with_device_and_scope(self, op_rec_column, create_new_out_column=None):
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
        if op_rec_column.already_sent is not False:
            raise RLGraphBuildError(
                "op_rec_column ID={} already sent through graph_fn '{}'! Cannot do so again.".format(
                    op_rec_column.id, op_rec_column.graph_fn.__name__)
            )

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
        op_rec_column.already_sent = True
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

    def run_through_graph_fn(self, op_rec_column, create_new_out_column=None):
        """
        Pushes all ops in the column through the respective graph_fn (graph_fn-spec and call-options are part of
        the column).
        Call options include flattening ops, flattening+splitting ops and (when splitting) adding the auto-generated
        flat key as first parameter to the different (split) calls of graph_fn.
        After the call, the already existing output column is populated with the actual results.

        Args:
            op_rec_column (DataOpRecordColumnIntoGraphFn): The column of DataOpRecords to be fed through the
                graph_fn.
            create_new_out_column (Optional[bool]): If given, whether to produce the out op-record column
                (or use the one already in the meta-graph). If True and the `op_rec_column` already links to an out
                op-rec column, raises an error.
                Default: None, meaning only create a new column if one dies not exist.

        Returns:
            DataOpRecordColumnFromGraphFn: The op-record column coming out of the graph_fn. This column may have
                already existed in the meta-graph before the graph_fn call or may have been generated during this
                call (if `create_new_out_column` is True).
        """
        args = [r.op for r in op_rec_column.op_records if r.kwarg is None]
        kwargs = {r.kwarg: r.op for r in op_rec_column.op_records if r.kwarg is not None}
        assert all(op is not None for op in args)  # just make sure

        call_time = None
        is_build_time = self.phase == "building"

        # Build the ops from this input-combination.
        # Flatten input items.
        # print("calling graph_fn = ", op_rec_column.graph_fn)
        # print("args = ", args)
        # print("kwargs = ", kwargs)
        if op_rec_column.flatten_ops is not False:
            flattened_args, flattened_kwargs = op_rec_column.flatten_input_ops(*args, **kwargs)
            # Split into SingleDataOps?
            if op_rec_column.split_ops:
                split_args_and_kwargs = op_rec_column.split_flattened_input_ops(*flattened_args, **flattened_kwargs)
                # There is some splitting to do. Call graph_fn many times (one for each split).
                if isinstance(split_args_and_kwargs, FlattenedDataOp):
                    ops = {}
                    num_return_values = -1
                    for key, params in split_args_and_kwargs.items():
                        params_args = params[0]
                        params_kwargs = params[1]
                        if is_build_time and TraceContext.ACTIVE_CALL_CONTEXT is False:
                            TraceContext.ACTIVE_CALL_CONTEXT = True
                            TraceContext.CONTEXT_START = time.perf_counter()

                        ops[key] = force_tuple(op_rec_column.graph_fn(op_rec_column.component,
                                                                      *params_args, **params_kwargs))
                        if is_build_time and TraceContext.ACTIVE_CALL_CONTEXT is True:
                            self.graph_call_times.append(time.perf_counter() - TraceContext.CONTEXT_START)
                            TraceContext.CONTEXT_START = None
                            TraceContext.ACTIVE_CALL_CONTEXT = False

                        if num_return_values >= 0 and num_return_values != len(ops[key]):
                            raise RLGraphError(
                                "Different split-runs through {} do not return the same number of values!".
                                format(op_rec_column.graph_fn.__name__)
                            )
                        num_return_values = len(ops[key])

                    # Un-split the results dict into a tuple of `num_return_values` slots.
                    un_split_ops = []
                    for i in range(num_return_values):
                        dict_with_singles = FlattenedDataOp()
                        for key in split_args_and_kwargs.keys():
                            dict_with_singles[key] = ops[key][i]
                        un_split_ops.append(dict_with_singles)
                    ops = tuple(un_split_ops)

                # No splitting to do: Pass everything as-is.
                else:
                    split_args, split_kwargs = split_args_and_kwargs[0], split_args_and_kwargs[1]
                    if is_build_time and TraceContext.ACTIVE_CALL_CONTEXT is False:
                        TraceContext.ACTIVE_CALL_CONTEXT = True
                        TraceContext.CONTEXT_START = time.perf_counter()
                    ops = op_rec_column.graph_fn(op_rec_column.component, *split_args, **split_kwargs)
                    if is_build_time and TraceContext.ACTIVE_CALL_CONTEXT is True:
                        self.graph_call_times.append(time.perf_counter() - TraceContext.CONTEXT_START)
                        TraceContext.CONTEXT_START = None
                        TraceContext.ACTIVE_CALL_CONTEXT = False
            else:
                if is_build_time and TraceContext.ACTIVE_CALL_CONTEXT is False:
                    TraceContext.ACTIVE_CALL_CONTEXT = True
                    TraceContext.CONTEXT_START = time.perf_counter()

                ops = op_rec_column.graph_fn(op_rec_column.component, *flattened_args, **flattened_kwargs)
                if is_build_time and TraceContext.ACTIVE_CALL_CONTEXT is True:
                    self.graph_call_times.append(time.perf_counter() - TraceContext.CONTEXT_START)
                    TraceContext.CONTEXT_START = None
                    TraceContext.ACTIVE_CALL_CONTEXT = False
        # Just pass in everything as-is.
        else:
            if is_build_time and TraceContext.ACTIVE_CALL_CONTEXT is False:
                call_time = time.perf_counter()
                TraceContext.ACTIVE_CALL_CONTEXT = True
                TraceContext.CONTEXT_START = call_time

            ops = op_rec_column.graph_fn(op_rec_column.component, *args, **kwargs)
            if is_build_time and TraceContext.ACTIVE_CALL_CONTEXT is True:
                self.graph_call_times.append(time.perf_counter() - TraceContext.CONTEXT_START)
                TraceContext.CONTEXT_START = None
                TraceContext.ACTIVE_CALL_CONTEXT = False
        # Make sure everything coming from a computation is always a tuple (for out-Socket indexing).
        ops = force_tuple(ops)

        # Always un-flatten all return values. Otherwise, we would allow Dict Spaces
        # with '/' keys in them, which is not allowed.
        ops = op_rec_column.unflatten_output_ops(*ops)

        # Should we create a new out op-rec column?
        if create_new_out_column is not False:
            # Assert that we don't have an out column already (wouldn't make sense).
            if create_new_out_column is True and op_rec_column.out_graph_fn_column is not None:
                raise RLGraphError(
                    "New DataOpRecordColumnFromGraphFn requested, but one already exists in in-column "
                    "{}!".format(op_rec_column)
                )
            if op_rec_column.out_graph_fn_column is None:
                out_graph_fn_column = DataOpRecordColumnFromGraphFn(
                    len(ops), component=op_rec_column.component, graph_fn_name=op_rec_column.graph_fn.__name__,
                    in_graph_fn_column=op_rec_column
                )
            else:
                out_graph_fn_column = op_rec_column.out_graph_fn_column
        else:
            assert op_rec_column.out_graph_fn_column is not None,\
                "ERROR: DataOpRecordColumnFromGraphFn for in-column {} is None!".format(op_rec_column)
            out_graph_fn_column = op_rec_column.out_graph_fn_column

        # Make sure the number of returned ops matches the number of op-records in the next column.
        # TODO: instead of backend check, do a build mode check here.
        # Define-by-run may return Nothing or None which is not an Op.
        if get_backend() == "tf":
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

    def sanity_check_build(self, still_building=False):
        """
        Checks whether some of the root component's API-method output columns contain ops that are still None.
        """
        for api_method_rec in self.root_component.api_methods.values():
            # Ignore `variables` API-method for now.
            if api_method_rec.name == "variables":
                continue
            for out_op_column in api_method_rec.out_op_columns:
                for op_rec in out_op_column.op_records:
                    if op_rec.op is None:
                        try:
                            self._analyze_none_op(op_rec)
                        except RLGraphBuildError as e:
                            if still_building:
                                print("Found problem in build process (causing a build-deadlock):")
                            raise e  # TODO: do something else with this error?

    def _analyze_none_op(self, op_rec):
        """
        Args:
            op_rec (DataOpRecord): The op-rec to analyze for errors (whose op property is None).

        Raises:
            RLGraphError: After the problem has been identified.
        """
        initial_op_rec = op_rec  # For debugging purposes.
        # Step via `previous` through the graph backwards.
        while True:
            previous_op_rec = op_rec.previous
            # Hit a graph_fn. Jump to incoming column.
            if previous_op_rec is None:
                # We have reached the beginning of the graph with a "variables:.."-dependent Space, so no op expected
                # yet.
                if isinstance(op_rec.space, str) and re.match(r'^variables:.+', op_rec.space):
                    assert False, "  Needs error message here!"

                else:
                    assert isinstance(op_rec.column, DataOpRecordColumnFromGraphFn),\
                        "ERROR: If previous op-rec is None, column must be of type `DataOpRecordColumnFromGraphFn` " \
                        "(but is type={})!".format(type(op_rec.column).__name__)
                    # All op-recs going into this graph_fn have actual ops.
                    # -> We know now that this graph_fn is only not called because the Component is either
                    # input-incomplete or variable-incomplete.
                    if op_rec.column.in_graph_fn_column.is_complete():
                        if op_rec.column.component.input_complete is False:
                            self._analyze_input_incomplete_component(op_rec.column.component)
                        else:
                            assert op_rec.column.component.variable_complete is False and \
                                   op_rec.column.in_graph_fn_column.requires_variable_completeness is True, \
                                   "ERROR: Component '{}' was expected to be either input-incomplete or " \
                                   "variable-incomplete!".format(op_rec.column.component.global_scope)
                            self._analyze_variable_incomplete_component(op_rec.column.component)
                    else:
                        # Take as an example None op the first incoming one that's None as well.
                        empty_in_op_recs = list(or_ for or_ in op_rec.column.in_graph_fn_column.op_records if or_.op is None)
                        if len(empty_in_op_recs) > 0:
                            previous_op_rec = empty_in_op_recs[0]
                        # All op-recs have actual ops -> .
                        else:
                            pass  # TODO: complete logic
            # Continue with new op-record.
            op_rec = previous_op_rec

    def _analyze_input_incomplete_component(self, component):
        """
        Analyzes why a component is input-incomplete and what we can further track back from it
        (e.g. maybe there is another one before that that is also input-incomplete).

        Args:
            component (Component): The defunct Component to analyze.

        Raises:
            RLGraphError: After the problem has been identified.
        """
        self.investigated_input_incomplete_components.add(component)
        # Find any input-param that has no Space defined.
        incomplete_input_args = list(name for name, space in component.api_method_inputs.items() if space is None)
        assert len(incomplete_input_args) > 0,\
            "ERROR: Expected at least one input-arg of '{}' to be without Space-definition!".\
            format(component.global_scope)
        incomplete_arg = incomplete_input_args[0]
        ## Either we are blocked by ourselves (API-method calling another one of the same Component (internal deadlock)).
        #internal_deadlock = False
        # Now loop through all of the Component's API-methods and look for all calls using `incomplete_arg`.
        # If they are all coming from the same Component -> internal deadlock.
        # If at least one is coming from another component -> follow that one via this very method.
        # TODO: What if there is external deadlock? Two components
        calls_using_incomplete_arg = 0
        components_making_calls_with_incomplete_arg = set()
        for api_method_name, api_method_rec in component.api_methods.items():
            # Found a call with `incomplete_arg`.
            if incomplete_arg in api_method_rec.input_names and len(api_method_rec.in_op_columns) > 0:
                calls_using_incomplete_arg += len(api_method_rec.in_op_columns)
                for call_column in api_method_rec.in_op_columns:
                    for op_rec in call_column.op_records:
                        assert op_rec.previous is not None
                        components_making_calls_with_incomplete_arg.add(op_rec.previous.column.component)

        must_be_complete_suggestion = \
            "If the space for this arg is not important in creating variables for this component, try flagging the " \
            "API-methods that use this arg via the `must_be_complete=False` flag."

        # What if incomplete_arg was never used in any calls?
        # Then that's the reason, why this component is input-incomplete.
        if calls_using_incomplete_arg == 0:
            raise RLGraphBuildError(
                "The call argument '{}' in Component '{}' was never used in any calls to any API-method of this "
                "component! Thus, the component remains input-incomplete. "
                "{}".format(incomplete_arg, component.global_scope, must_be_complete_suggestion)
            )
        # Only this very component uses this call arg -> "Inner deadlock".
        elif len(components_making_calls_with_incomplete_arg) == 1 and \
                component in components_making_calls_with_incomplete_arg:
            raise RLGraphBuildError(
                "Component '{}' has a circular dependency via API call arg '{}'! Only this component ever makes "
                "calls using this arg, so it can never become input-complete. "
                "{}".format(component.global_scope, incomplete_arg, must_be_complete_suggestion)
            )
        # Some other component(s) use this call arg, but they might be input-incomplete themselves.
        else:
            for calling_component in components_making_calls_with_incomplete_arg:
                if calling_component is component:
                    continue
                # Assume that the caller is input-incomplete itself.
                assert calling_component.input_complete is False
                # Continue investigating why this one is input incomplete.

    def _analyze_variable_incomplete_component(self, component):
        """
        Analyzes why a component is variable-incomplete (one of its children is not input-complete) and keeps tracking
        the root cause for this problem.

        Args:
            component (Component): The defunct Component to analyze.

        Raises:
            RLGraphError: After the problem has been identified.
        """
        # Find the sub-component that's input-incomplete and further track that one.
        for sub_component in component.get_all_sub_components():
            if sub_component.input_complete is False:
                self._analyze_input_incomplete_component(sub_component)

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

            api_method_name = api_method_call
            params = []
            return_ops = None

            # Call is defined by a list/tuple of [method], [input params], [return_ops]?
            if isinstance(api_method_call, (list, tuple)):
                api_method_name = api_method_call[0] if not callable(api_method_call[0]) else \
                    api_method_call[0].__name__
                # If input is one dict: Check first placeholder for being a dict as well and if so, do a normal 1:1
                # mapping, otherwise, roll out the input dict as a list.
                if isinstance(api_method_call[1], dict) and \
                        not isinstance(self.api[api_method_name][0][0].op, DataOpDict):
                    params = [v for k, v in sorted(api_method_call[1].items())]
                else:
                    params = force_list(api_method_call[1])

                return_ops = force_list(api_method_call[2]) if len(api_method_call) > 2 and \
                                                               api_method_call[2] is not None else None
            # Allow passing the function directly
            if callable(api_method_call):
                api_method_name = api_method_call.__name__

            if api_method_name not in self.api:
                raise RLGraphError("No API-method with name '{}' found!".format(api_method_name))

            # API returns a dict.
            if len(self.api[api_method_name][1]) > 0 and self.api[api_method_name][1][0].kwarg is not None:
                for op_rec in self.api[api_method_name][1]:
                    if return_ops is None or op_rec.kwarg in return_ops:
                        if api_method_name not in fetch_dict:
                            fetch_dict[api_method_name] = {}
                        flat_ops = flatten_op(op_rec.op, mapping=lambda o: o.op if isinstance(o, DataOpRecord) else o)
                        fetch_dict[api_method_name][op_rec.kwarg] = unflatten_op(flat_ops)

                if return_ops is not None:
                    assert all(op in fetch_dict[api_method_name] for op in return_ops),\
                        "ERROR: Not all wanted return_ops ({}) are returned by API-method `api_method_call`!".format(
                        return_ops)
            # API returns a tuple.
            else:
                fetch_dict[api_method_name] = [op_rec.op for i, op_rec in enumerate(self.api[api_method_name][1]) if
                                               return_ops is None or i in return_ops]
                if return_ops is not None:
                    assert len(fetch_dict[api_method_name]) == len(return_ops),\
                        "ERROR: Not all wanted return_ops ({}) are returned by API-method `api_method_call`!".format(
                        return_ops)

            for i, param in enumerate(params):
                if param is None:
                    assert len(self.api[api_method_name][0]) == i, \
                        "ERROR: More input params given ({}) than expected ({}) for call to '{}'!". \
                        format(len(params), len(self.api[api_method_name][0]), api_method_name)
                    break

                # TODO: What if len(params) < len(self.api[api_method][0])?
                # Need to handle default API-method params also for the root-component (this one).
                if len(self.api[api_method_name][0]) <= i:
                    raise RLGraphError(
                        "API-method with name '{}' only has {} input parameters! You passed in "
                        "{}.".format(api_method_name, len(self.api[api_method_name][0]), len(params))
                    )

                placeholder = self.api[api_method_name][0][i].op  # 0=input op-recs; i=ith input op-rec
                if isinstance(placeholder, ContainerDataOp):
                    flat_placeholders = flatten_op(placeholder)
                    for flat_key, value in flatten_op(param).items():
                        feed_dict[flat_placeholders[flat_key]] = value
                # Special case: Get the default argument for this arg.
                # TODO: Support API-method's kwargs here as well (mostly useful for test.test).
                #elif param is None:
                #    feed_dict[placeholder] = self.root_component.api_methods[api_method_call].default_values[i]
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

    def execute_define_by_run_graph_fn(self, component, graph_fn, options, *args, **kwargs):
        """
        Executes a graph_fn in define by run mode.

        Args:
            component (Component): Component this graph_fn is eecuted on.
            graph_fn (callable): Graph function to execute.
            options (dict): Execution options.
        Returns:
            any: Results of executing this graph-fn.
        """
        flatten_ops = options.pop("flatten_ops", False)
        split_ops = options.pop("split_ops", False)
        add_auto_key_as_first_param = options.pop("add_auto_key_as_first_param", False)

        # No container arg handling.
        if not flatten_ops:
            return graph_fn(component, *args, **kwargs)
        else:
            # Flatten and identify containers for potential splits.
            flattened_args = []

            # Was there actually any flattening
            args_actually_flattened = False
            for arg in args:
                if isinstance(arg, (Dict, dict, tuple)) or isinstance(arg, Dict) or isinstance(arg, tuple):
                    flattened_args.append(define_by_run_flatten(arg))
                    args_actually_flattened = True
                else:
                    flattened_args.append(arg)

            flattened_kwargs = {}
            if len(kwargs) > 0:
                for key, arg in kwargs.items():
                    if isinstance(arg, dict) or isinstance(arg, Dict) or isinstance(arg, tuple):
                        flattened_kwargs[key] = define_by_run_flatten(arg)
                        args_actually_flattened = True
                    else:
                        flattened_kwargs[key] = arg

            # If splitting args, split then iterate and merge. Only split if some args were actually flattened.
            if args_actually_flattened:
                split_args_and_kwargs = define_by_run_split_args(add_auto_key_as_first_param,
                                                                 *flattened_args, **flattened_kwargs)

                # Idea: Unwrap light flattening by iterating over flattened args and reading out "" where possible
                if split_ops and isinstance(split_args_and_kwargs, OrderedDict):
                    # Args were actually split.
                    ops = {}
                    num_return_values = -1
                    for key, params in split_args_and_kwargs.items():
                        # Are there any kwargs?
                        if isinstance(params, tuple):
                            params_args = params[0]
                            params_kwargs = params[1]
                        else:
                            params_args = params
                            params_kwargs = {}
                        ops[key] = graph_fn(component, *params_args, **params_kwargs)
                        if hasattr(ops[key], "shape"):
                            num_return_values = 1
                        else:
                            num_return_values = len(ops[key])

                    # Un-split the results dict into a tuple of `num_return_values` slots.
                    un_split_ops = []
                    for i in range(num_return_values):
                        dict_with_singles = OrderedDict()
                        for key in split_args_and_kwargs.keys():
                            # Use tensor as is.
                            if hasattr(ops[key], "shape"):
                                dict_with_singles[key] = ops[key]
                            else:
                                dict_with_singles[key] = ops[key][i]
                        un_split_ops.append(dict_with_singles)

                    flattened_ret = tuple(un_split_ops)
                else:
                    if isinstance(split_args_and_kwargs, OrderedDict):
                        flattened_ret = graph_fn(component, split_args_and_kwargs)
                    else:
                        # Args and kwargs tuple.
                        split_args = split_args_and_kwargs[0]
                        split_kwargs = split_args_and_kwargs[1]
                        # Args did not contain deep nested structure so
                        flattened_ret = graph_fn(component, *split_args, **split_kwargs)

                # If result is a raw tensor, return as is.
                if get_backend() == "pytorch":
                    if isinstance(flattened_ret, torch.Tensor):
                        return flattened_ret

                unflattened_ret = []
                for i, op in enumerate(flattened_ret):
                    # Try to re-nest ordered-dict it.
                    if isinstance(op, OrderedDict):
                        unflattened_ret.append(define_by_run_unflatten(op))
                    # All others are left as-is.
                    else:
                        unflattened_ret.append(op)

                # Return unflattened results.
                return unflattened_ret[0] if len(unflattened_ret) == 1 else unflattened_ret
            else:
                # Just pass in args and kwargs because not actually flattened, with or without default key.
                if add_auto_key_as_first_param:
                    ret = graph_fn(component, "", *args, **kwargs)
                else:
                    ret = graph_fn(component, *args, **kwargs)
                return define_by_run_unpack(ret)

    def build_define_by_run_graph(self, meta_graph, input_spaces, available_devices,
                                  device_strategy="default", default_device=None, device_map=None):
        """
        Builds a graph for eager or define by run execution. This primarily consists of creating variables through
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
        self.phase = "building"
        TraceContext.DEFINE_BY_RUN_CONTEXT = "building"

        # TODO device strategy in pytorch?
        # Build full registry of callable methods on root component.
        for member in inspect.getmembers(self.root_component):
            name, method = (member[0], member[1])

            # N.b. this means _graph_fns are not directly callable here, just api functions.
            if name not in self.root_component.api_fn_by_name and name in self.api:
                self.root_component.api_fn_by_name[name] = method

        # Create the first actual ops based on the input-spaces.
        # Some ops can only be created later when variable-based-Spaces are known (op_recs_depending_on_variables).
        self.build_input_space_ops(input_spaces)

        # Collect all components and add those op-recs to the set that are constant.
        components = self.root_component.get_all_sub_components()
        for component in components:
            component.graph_builder = self  # point to us.
            self.op_records_to_process.update(component.constant_op_records)
            # Check whether the Component is input-complete (and build already if it is).
            self.build_component_when_input_complete(component)

        op_records_list = self._sort_op_recs(self.op_records_to_process)
        iterations = self._build(op_records_list)

        # Set execution mode in components to change `call` behaviour to direct function evaluation.
        self.root_component.propagate_sub_component_properties(properties=dict(execution_mode="define_by_run"))

        # Call post build logic.
        self.root_component._post_build(self.root_component)

        time_build = time.perf_counter() - time_start
        self.logger.info("Define-by-run computation-graph build completed in {} s ({} iterations).".
                         format(time_build, iterations))
        build_overhead = time_build - sum(self.graph_call_times) - sum(self.var_call_times)
        TraceContext.DEFINE_BY_RUN_CONTEXT = "execution"
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
            # In this iteration, do we still have API-method op-recs (which are part of columns that go into or come
            # from API-methods).
            have_api_method_recs = any(
                isinstance(or_.column, (DataOpRecordColumnIntoAPIMethod, DataOpRecordColumnFromAPIMethod)) for or_ in
                op_records_list
            )
            # Keep track of the highest nesting-level (depth of a component in the parent/child-component-tree) for
            # a called graph_fn. Graph_fns with a lower nesting level will not be called in the same iteration.
            # This allows for careful progress through the graph_fn-calls in case API methods of
            # input/variable-incomplete components are called within these graph_fns (to be avoided at all costs
            # as it will fail the build).
            highest_nesting_of_called_graph_fn_column = -1

            # Collect op-recs to process in the next iteration.
            self.op_records_to_process = set()

            # Set of Components that have been tried last to get input-complete. If build gets stuck, it'll be because
            # of the Components in this set.
            non_complete_components = set()
            for op_rec in op_records_list:  # type: DataOpRecord
                # There are next records:
                if len(op_rec.next) > 0:
                    # Push actual op and Space forward one op-rec at a time.
                    for next_op_rec in self._sort_op_recs(op_rec.next):  # type: DataOpRecord
                        # Assert that next-record's `previous` field points back to op_rec.
                        assert next_op_rec.previous is op_rec, \
                            "ERROR: Op-rec {} in meta-graph has {} as next, but {}'s previous field points to {}!". \
                            format(op_rec, next_op_rec, next_op_rec, next_op_rec.previous)
                        # If not last op in this API-method -> continue.
                        if next_op_rec.is_terminal_op is False:
                            assert next_op_rec.op is None or is_constant(next_op_rec.op) or next_op_rec.op is op_rec.op
                            self.op_records_to_process.add(next_op_rec)
                        # Push op and Space into next op-record.
                        # With op-instructions?
                        if "key-lookup" in next_op_rec.op_instructions:
                            lookup_key = next_op_rec.op_instructions["key-lookup"]
                            if isinstance(lookup_key, str) and (not isinstance(op_rec.op, dict) or lookup_key
                                                                not in op_rec.op):
                                raise RLGraphError(
                                    "op_rec.op ({}) is not a dict or does not contain the lookup key '{}'!". \
                                    format(op_rec.op, lookup_key)
                                )
                            elif isinstance(lookup_key, int) and (not isinstance(op_rec.op, (list, tuple)) or
                                                                  lookup_key >= len(op_rec.op)):
                                raise RLGraphError(
                                    "op_rec.op ({}) is not a list/tuple or contains not enough items for lookup "
                                    "index '{}'!".format(op_rec.op, lookup_key)
                                )
                            next_op_rec.op = op_rec.op[lookup_key]
                            next_op_rec.space = op_rec.space[lookup_key]
                        # No instructions -> simply pass on.
                        else:
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
                                # For non-space agnostic Components: Sanity check, whether Spaces are equivalent.
                                elif component.space_agnostic is False:
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
                                    next_op_rec.space = component.api_method_inputs[param_name] = generic_space

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
                    # Only call the GraphFn iff:
                    # There are no more DataOpRecordColumnIntoAPIMethod ops in our list: We would like to hold off
                    # any graph fn calls for as long as possible.
                    # We don't want to run through a graph_fn, then have to call an API-method from within that graph_fn
                    # and the component of that API-method is not input-/variable-complete yet.
                    if have_api_method_recs:
                        # Recycle this op-rec.
                        self.op_records_to_process.add(op_rec)
                    # There are other graph_fn columns that have a higher Component nesting_level and are
                    # actually callable -> Call those first.
                    elif highest_nesting_of_called_graph_fn_column > op_rec.column.component.nesting_level:
                        # Recycle this op-rec.
                        self.op_records_to_process.add(op_rec)
                    # GraphFn column must be complete AND has not been sent through the graph_fn yet.
                    elif op_rec.column.is_complete() and op_rec.column.already_sent is False:
                        do_call = False  # Do the actual graph_fn call?

                        # Only call the graph_fn if the Component is already input-complete.
                        if op_rec.column.component.variable_complete or \
                                (op_rec.column.requires_variable_completeness is False and
                                 op_rec.column.component.input_complete):
                            do_call = True
                        # Component not input-/variable-complete yet. Recycle this op-rec.
                        else:
                            self.build_component_when_input_complete(op_rec.column.component)
                            self.op_records_to_process.add(op_rec)
                            if op_rec.column.component.input_complete is False:
                                non_complete_components.add(op_rec.column.component.global_scope)
                            # Call the graph_fn here right away iff component is ready now.
                            elif op_rec.column.component.variable_complete or \
                                    op_rec.column.requires_variable_completeness is False:
                                do_call = True

                        if do_call:
                            # Call the graph_fn with the given column and call-options.
                            self.run_through_graph_fn_with_device_and_scope(op_rec.column)
                            # Store all resulting op_recs (returned by the graph_fn) to be processed next.
                            self.op_records_to_process.update(op_rec.column.out_graph_fn_column.op_records)
                            highest_nesting_of_called_graph_fn_column = op_rec.column.component.nesting_level

                    # - There are still into-API-method-op-recs that should be handled first.
                    # - Op column is not complete yet: Discard this one (as others will keep coming in anyway).
                    # - Op column has already been sent (sibling ops may have arrived in same iteration).
                # else: - Op belongs to a column coming from a graph_fn or an API-method, but the op is no longer used.
                # -> Ignore Op.

            # If we are done with the build, check for API-methods' ops that are dependent on variables
            # generated during the build and build these now.
            # TODO is this loop necessary for define by run?
            if get_backend() == "tf":
                if len(self.op_recs_depending_on_variables) > 0:
                    op_records_list = list(self.op_recs_depending_on_variables)
                    self.op_recs_depending_on_variables = set()

                    # Loop through the op_records list and sanity check for "variables"-dependent Spaces, then get these
                    # Spaces (iff respective component is input-complete), create the placeholders and keep building.
                    for op_rec in op_records_list:
                        space_desc = op_rec.space  # type: str
                        mo = re.search(r'^variables:(.+)', space_desc)
                        assert mo
                        component_path = mo.group(1).split("/")
                        component = self.root_component
                        for level in component_path:
                            assert level in component.sub_components, \
                                "ERROR: `component_path` ('{}') contains non-existent Components!".format(
                                    component_path)
                            component = component.sub_components[level]
                        if component.variable_complete is True:
                            var_space = Dict({key: get_space_from_op(value) for key, value in sorted(
                                component.get_variables(custom_scope_separator="-").items()
                            )})
                            op_rec.space = var_space
                            placeholder_name = next(iter(op_rec.next)).column.api_method_rec.input_names[op_rec.position]
                            assert len(op_rec.next) == 1, \
                                "ERROR: root_component API op-rec ('{}') expected to have only one `next` op-rec!". \
                                format(placeholder_name)
                            op_rec.op = self.get_placeholder(
                                placeholder_name, space=var_space, component=self.root_component
                            )
                            self.op_records_to_process.add(op_rec)
                        else:
                            self.op_recs_depending_on_variables.add(op_rec)

            # Sanity check, whether we are stuck.
            new_op_records_list = self._sort_op_recs(self.op_records_to_process)
            if op_records_list == new_op_records_list:
                # Probably deadlocked. Do a premature sanity check to report possible problems.
                if loop_counter > self.max_build_iterations:
                    self.sanity_check_build(still_building=True)
                    return

            op_records_list = new_op_records_list

            loop_counter += 1
        return loop_counter

    @staticmethod
    def _sort_op_recs(recs):
        """
        Sorts op-recs according to:
        - Give API-method calls priority over GraphFn calls (API-method call ops just have to be passed along without
        worrying about input-/variable-completeness).
        - Give deeper nested Components priority over shallower nested ones.
        - Sort by op-rec ID to enforce determinism.

        Note: We sort in reverse order, highest key-values first.

        Args:
            recs (Set[DataOpRecord]): The DataOpRecords to sort.

        Returns:
            list: The sorted op-recs.
        """
        def sorting_func(rec):
            # Op-rec is a placeholder. Highest priority.
            if rec.column is None:
                return DataOpRecord.MAX_ID * 2 + rec.id
            # API-methods have priority (over GraphFns).
            elif isinstance(rec.column, DataOpRecordColumnIntoAPIMethod):
                return DataOpRecord.MAX_ID + rec.id
            # Deeper nested Components have priority. If same level, use op-rec's ID for determinism.
            return rec.column.component.nesting_level + rec.id / DataOpRecord.MAX_ID

        return sorted(recs, key=sorting_func, reverse=True)
