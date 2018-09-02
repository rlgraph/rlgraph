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
from six.moves import xrange as range_

import copy
import inspect
import numpy as np
import re

from rlgraph import get_backend
from rlgraph.utils.rlgraph_error import RLGraphError
from rlgraph.utils.specifiable import Specifiable

from rlgraph.utils.ops import SingleDataOp, DataOpDict, DataOpRecord, APIMethodRecord, \
    DataOpRecordColumnIntoGraphFn, DataOpRecordColumnFromGraphFn, DataOpRecordColumnIntoAPIMethod, \
    DataOpRecordColumnFromAPIMethod, GraphFnRecord, DataOpTuple, FLAT_TUPLE_OPEN, FLAT_TUPLE_CLOSE
from rlgraph.utils import util, default_dict
from rlgraph.spaces.space_utils import get_space_from_op

if get_backend() == "tf":
    import tensorflow as tf
elif get_backend() == "tf-eager":
    import tensorflow as tf
    import tensorflow.contrib.eager as eager


class Component(Specifiable):
    """
    Base class for a graph component (such as a layer, an entire function approximator, a memory, an optimizers, etc..).

    A component can contain other components and/or its own graph-logic (e.g. tf ops).
    A component's sub-components are connected to each other via in- and out-Sockets (similar to LEGO blocks
    and deepmind's sonnet).

    This base class implements the interface to add sub-components, create connections between
    different sub-components and between a sub-component and this one and between this component
     and an external component.

    A component also has a variable registry, the ability to save the component's structure and variable-values to disk,
    and supports adding its graph_fns to the overall computation graph.
    """

    def __init__(self, *sub_components, **kwargs):
        """
        Args:
            sub_components (Component): Specification dicts for sub-Components to be added to this one.

        Keyword Args:
            name (str): The name of this Component. Names of sub-components within a containing component
                must be unique. Names are used to label exposed Sockets of the containing component.
                If name is empty, use scope as name (as last resort).
            scope (str): The scope of this Component for naming variables in the Graph.
            device (str): Device this component will be assigned to. If None, defaults to CPU.
            trainable (Optional[bool]): Whether to make the variables of this Component always trainable or not.
                Use None for no specific preference.
            #global_component (bool): In distributed mode, this flag indicates if the component is part of the
            #    shared global model or local to the worker. Defaults to False and will be ignored if set to
            #    True in non-distributed mode.

            # TODO: remove when we have numpy-based Components (then we can do test calls to infer everything automatically)
            graph_fn_num_outputs (dict): A dict specifying which graph_fns have how many return values.
                This can be useful if graph_fns don't clearly have a fixed number of return values and the auto-inferral
                utility function cannot determine the actual number of returned values.

            switched_off_apis (Optional[Set[str]]): Set of API-method names that should NOT be build for this Component.
            backend (str): The custom backend that this Component obliges to. None to use the RLGraph global backend.
                Default: None.
        """
        super(Component, self).__init__()

        # Scope if used to create scope hierarchies inside the Graph.
        # self.logger = logging.getLogger(__name__)
        self.scope = kwargs.pop("scope", "")

        assert re.match(r'^[\w\-]*$', self.scope), \
            "ERROR: scope {} does not match scope-pattern! Needs to be \\w or '-'.".format(self.scope)
        # The global scope string defining the exact nested position of this Component in the Graph.
        # e.g. "/core/component1/sub-component-a"
        self.global_scope = self.scope

        # Shared variable scope.
        self.reuse_variable_scope = kwargs.pop("reuse_variable_scope", None)

        # Names of sub-components that exist (parallelly) inside a containing component must be unique.
        self.name = kwargs.pop("name", self.scope)  # if no name given, use scope
        self.device = kwargs.pop("device", None)
        self.trainable = kwargs.pop("trainable", None)
        #self.global_component = kwargs.pop("global_component", False)
        self.graph_fn_num_outputs = kwargs.pop("graph_fn_num_outputs", dict())
        self.switched_off_apis = kwargs.pop("switched_off_apis", set())
        self.backend = kwargs.pop("backend", None)

        assert not kwargs, "ERROR: kwargs ({}) still contains items!".format(kwargs)

        # Keep track of whether this Component has already been added to another Component and throw error
        # if this is done twice. Each Component can only be added once to a parent Component.
        self.parent_component = None  # type: Component

        # Dict of sub-components that live inside this one (key=sub-component's scope).
        self.sub_components = OrderedDict()

        # Link to the GraphBuilder object.
        self.graph_builder = None

        # `self.api_methods`: Dict holding information about which op-record-tuples go via which API
        # methods into this Component and come out of it.
        # keys=API method name; values=APIMethodRecord
        self.api_methods = dict()

        # Maps names to callable API functions for eager calls.
        self.api_fn_by_name = dict()
        # Maps names of methods to synthetically defined methods.
        self.synthetic_methods = set()

        # How this component executes its 'call' method.
        self.execution_mode = "static_graph"

        # `self.api_method_inputs`: Registry for all unique API-method input parameter names and their Spaces.
        # Two API-methods may share the same input if their input parameters have the same names.
        # keys=input parameter name; values=Space that goes into that parameter
        self.api_method_inputs = dict()
        self.get_api_methods()

        # Registry for graph_fn records (only populated at build time when the graph_fns are actually called).
        self.graph_fns = dict()
        # Set of op-rec-columns going into a graph_fn of this Component and not having 0 op-records.
        # Helps during the build procedure to call these right away after the Component is input-complete.
        self.no_input_graph_fn_columns = set()
        # Set of op-records that are constant and thus can be processed right away at the beginning of the build
        # procedure.
        self.constant_op_records = set()
        # Whether we know already all our in-Sockets' Spaces.
        # Only then can we create our variables. Model will do this.
        self.input_complete = False
        # Whether all our sub-Components are input-complete. Only after that point, we can run our _variables graph_fn.
        self.variable_complete = False

        # All Variables that are held by this component (and its sub-components) by name.
        # key=full-scope variable name (scope=component/sub-component scope)
        # value=the actual variable
        self.variables = dict()
        # All summary ops that are held by this component (and its sub-components) by name.
        # key=full-scope summary name (scope=component/sub-component scope)
        # value=the actual summary op
        self.summaries = dict()
        # The regexp that a summary's full-scope name has to match in order for it to be generated and registered.
        # This will be set by the GraphBuilder at build time.
        self.summary_regexp = None

        # Now add all sub-Components (also support all sub-Components being given in a single list).
        sub_components = sub_components[0] if len(sub_components) == 1 and \
            isinstance(sub_components[0], (list, tuple)) else sub_components
        self.add_components(*sub_components)

        # Define the "_variables" API-method that each Component automatically has.
        self.define_api_method("_variables", self._graph_fn__variables)

    def get_api_methods(self):
        """
        Detects all methods of the Component that should be registered as API-methods for
        this Component and complements `self.api_methods` and `self.api_method_inputs`.
        #Returns:
        #    Two dicts:
        #        - Dict[str,APIMethodRecord]: Dict of key=API-method name (str); values=APIMethodRecord.
        #        - Dict[str,Space]: Dict of key=API-method input parameter name (str); values=Space.
        """
        #api_method_records = dict()
        #api_method_inputs = dict()
        # Look for all our API methods (those that use the `call` method).
        # TODO use dict of callables -> api_fn_by_name
        for member in inspect.getmembers(self):
            name, method = (member[0], member[1])
            if name != "define_api_method" and name != "add_components" and name[0] != "_" and \
                    name not in self.switched_off_apis and callable(method) and util.get_method_type(method) == "api":
                # callable_anytime = False  # not util.does_method_call_graph_fns(method)
                self.api_methods[name] = APIMethodRecord(method, component=self)  #, callable_anytime=callable_anytime)
                # Store the input-parameter names and map them to None records (Space not defined yet).
                param_list = inspect.signature(method).parameters.values()
                self._complement_api_method_registries(name, param_list)

                # api_method_records[name].input_names = [p.name for p in params]
                # for param in params:
                #    if param.name not in api_method_inputs:
                #        TODO: write function for this block and the corresponding one in `define_api_method`.
                #        api_method_inputs[param.name] = None
        # return api_method_records, api_method_inputs

    def call(self, method, *params, **kwargs):
        """
        Performs either:
        a) An assembly run through another API method (will actually call this API method for further assembly).
        b) A dry run through a graph_fn (without calling it) just generating the empty op-record-columns around the
            graph_fn (incoming and outgoing).

        Args:
            method (callable): The method (graph_fn or API method) to call.
            *params (Union[DataOpRecord,DataOp]): The DataOpRecords/DataOps to be used for calling the method.

        Keyword Args:
            flatten_ops (Union[bool,Set[str]]): Whether to flatten all or some DataOps by creating
                a FlattenedDataOp (with automatic key names).
                Can also be a set of in-Socket names to flatten explicitly (True for all).
                (default: True).
            split_ops (Union[bool,Set[str]]): Whether to split all or some of the already flattened DataOps
                and send the SingleDataOps one by one through the graph_fn.
                Example: Spaces=A=Dict (container), B=int (primitive)
                    The graph_fn should then expect for each primitive Space in A:
                        _graph_fn(primitive-in-A (Space), B (int))
                        NOTE that B will be the same in all calls for all primitive-in-A's.
                (default: True).
            add_auto_key_as_first_param (bool): If `split_ops` is not False, whether to send the
                automatically generated flat key as the very first parameter into each call of the graph_fn.
                Example: Spaces=A=float (primitive), B=Tuple (container)
                    The graph_fn should then expect for each primitive Space in B:
                        _graph_fn(key, A (float), primitive-in-B (Space))
                        NOTE that A will be the same in all calls for all primitive-in-B's.
                        The key can now be used to index into variables equally structured as B.
                Has no effect if `split_ops` is False.
                (default: False).
            return_ops (bool): Whether to return actual ops rather than op-records. This is done automatically
                (regardless of this value), if the direct parent caller of this method is a `_graph_fn_`-method.
                Default: False.

        Returns:
            Tuple[DataOpRecord]: The returned tuple of DataOpRecords coming from the called API-method or graph_fn.
        """
        # Owner of method:
        method_owner = method.__self__  # type: Component
        # Check, whether the method-owner is either this Component or has this Component as parent.
        # TODO: make this check generic for any depth-grand-child.A
        assert method_owner is self or self in method_owner.get_parents(), \
            "ERROR: Can only call API-method ({}/{}) on self ({}) or any sub-Components of self! Most likely, " \
            "{} has not been added to self.". \
            format(method_owner.global_scope, method.__name__, self.global_scope, method_owner.global_scope)
        # Try handing the graph-builder link to the owner Component (in case it's not set yet).
        if method_owner.graph_builder is None:
            method_owner.graph_builder = self.graph_builder

        return_ops = kwargs.pop("return_ops", False)

        # Direct evaluation of function.
        if self.execution_mode == "define_by_run":
            return method(*params, **kwargs)
        elif self.execution_mode == "static_graph":
            # Graph construction.

            # Method is an API method.
            if method.__name__ in method_owner.api_methods:
                # Make the API call.
                op_recs = self.call_api(method, method_owner, *params, **kwargs)

            # Method is a graph_fn.
            else:
                op_recs = self.call_graph_fn(method, method_owner, *params, **kwargs)

        # Do we need to return the raw ops or the op-recs?
        # Direct parent caller is a `_graph_fn_...`: Return raw ops.
        stack = inspect.stack()
        if return_ops is True or re.match(r'^_graph_fn_.+$', stack[1][3]):
            return tuple(o.op for o in op_recs) if isinstance(op_recs, tuple) else op_recs.op
        # Parent caller is non-graph_fn: Return op-recs.
        else:
            return op_recs

    def call_graph_fn(self, method, method_owner, *params, **kwargs):
        """
        Executes a dry run through a graph_fn (without calling it) just generating the empty
        op-record-columns around the graph_fn (incoming and outgoing).

        Args:
            method (callable): The method (graph_fn or API method) to call.
            method_owner (Component): Component this method belongs to.
            *params (Union[DataOpRecord,np.array,numeric]): The DataOpRecords to be used for calling the method.:

        Keyword Args:
            flatten_ops (Union[bool,Set[str]]): Whether to flatten all or some DataOps by creating
                a FlattenedDataOp (with automatic key names).
                Can also be a set of in-Socket names to flatten explicitly (True for all).
                (default: True).
            split_ops (Union[bool,Set[str]]): Whether to split all or some of the already flattened DataOps
                and send the SingleDataOps one by one through the graph_fn.
                Example: Spaces=A=Dict (container), B=int (primitive)
                    The graph_fn should then expect for each primitive Space in A:
                        _graph_fn(primitive-in-A (Space), B (int))
                        NOTE that B will be the same in all calls for all primitive-in-A's.
                (default: True).
            add_auto_key_as_first_param (bool): If `split_ops` is not False, whether to send the
                automatically generated flat key as the very first parameter into each call of the graph_fn.
                Example: Spaces=A=float (primitive), B=Tuple (container)
                    The graph_fn should then expect for each primitive Space in B:
                        _graph_fn(key, A (float), primitive-in-B (Space))
                        NOTE that A will be the same in all calls for all primitive-in-B's.
                        The key can now be used to index into variables equally structured as B.
                Has no effect if `split_ops` is False.
                (default: False).

        """
        flatten_ops = kwargs.pop("flatten_ops", False)
        split_ops = kwargs.pop("split_ops", False)
        add_auto_key_as_first_param = kwargs.pop("add_auto_key_as_first_param", False)

        # Make sure the graph_fn belongs to this Component (not allowed to call graph_fn of other component
        # directly).
        if method_owner is not self:
            raise RLGraphError("Graph_fn '{}' may only be sent to `call` by its owner ({})! However, '{}' is "
                               "calling it.".format(method.__name__, method_owner.scope, self.scope))

        # Add all kwargs to params (op-rec contains kwarg-name).
        params = list(params)
        for key, value in kwargs.items():
            value.kwarg = key
            params.append(value)

        # Sanity check number of actual graph_fn input values against len(params).
        self.sanity_check_call_parameters(params, method, "graph_fn", add_auto_key_as_first_param)

        # Store a  graph_fn record in this component for better in/out-op-record-column reference.
        if method.__name__ not in self.graph_fns:
            self.graph_fns[method.__name__] = GraphFnRecord(graph_fn=method, component=self)

        # Generate in-going op-rec-column.
        in_graph_fn_column = DataOpRecordColumnIntoGraphFn(
            len(params), component=self, graph_fn=method,
            flatten_ops=flatten_ops, split_ops=split_ops, add_auto_key_as_first_param=add_auto_key_as_first_param
        )
        # Add the column to the `graph_fns` record.
        self.graph_fns[method.__name__].in_op_columns.append(in_graph_fn_column)

        # We are already building: Actually call the graph_fn after asserting that its Component is input-complete.
        if self.graph_builder and self.graph_builder.phase == "building":
            # Populate in-op-column with actual ops, Space, kwarg-name.
            for i, in_op in enumerate(params):
                if isinstance(in_op, DataOpRecord):
                    in_graph_fn_column.op_records[i].op = in_op.op
                    in_graph_fn_column.op_records[i].space = get_space_from_op(in_op.op)
                    in_graph_fn_column.op_records[i].kwarg = in_op.kwarg
                elif in_op is not None:
                    in_graph_fn_column.op_records[i].op = in_op
                    in_graph_fn_column.op_records[i].space = get_space_from_op(in_op)
            # Assert input-completeness of Component (if not already, then after this graph_fn/Space update).
            if self.input_complete is False:
                # Check Spaces and create variables.
                self.graph_builder.build_component_when_input_complete(self)
                assert self.input_complete

            # Call the graph_fn.
            out_graph_fn_column = self.graph_builder.run_through_graph_fn_with_device_and_scope(
                in_graph_fn_column, create_new_out_column=True
            )

        # We are still in the assembly phase: Don't actually call the graph_fn. Only generate op-rec-columns
        # around it (in-coming and out-going).
        else:
            # Create 2 op-record columns, one going into the graph_fn and one getting out of there and link
            # them together via the graph_fn (w/o calling it).
            # TODO: remove when we have numpy-based Components (then we can do test calls to infer everything automatically)
            if method.__name__ in self.graph_fn_num_outputs:
                num_graph_fn_return_values = self.graph_fn_num_outputs[method.__name__]
            else:
                num_graph_fn_return_values = util.get_num_return_values(method)
            self.logger.debug("Graph_fn has {} return values (inferred).".format(method.__name__,
                                                                                 num_graph_fn_return_values))
            # If in-column is empty, add it to the "empty in-column" set.
            if len(in_graph_fn_column.op_records) == 0:
                self.no_input_graph_fn_columns.add(in_graph_fn_column)

            # Generate the out-op-column from the number of return values (guessed during assembly phase or
            # actually measured during build phase).
            out_graph_fn_column = DataOpRecordColumnFromGraphFn(
                num_graph_fn_return_values, component=self, graph_fn_name=method.__name__,
                in_graph_fn_column=in_graph_fn_column
            )

            in_graph_fn_column.out_graph_fn_column = out_graph_fn_column

        self.graph_fns[method.__name__].out_op_columns.append(out_graph_fn_column)

        # Link from in-going op-recs with out-coming ones (both ways).
        for i, op_rec in enumerate(params):
            # A DataOpRecord: Link to next and from next back to op_rec.
            if isinstance(op_rec, DataOpRecord):
                op_rec.next.add(in_graph_fn_column.op_records[i])
                in_graph_fn_column.op_records[i].previous = op_rec
                # Also update the kwarg name so that a future call to the graph_fn with the in-column will work.
                in_graph_fn_column.op_records[i].kwarg = op_rec.kwarg
            # A fixed input value. Store directly as op with Space and register it as already known (constant).
            elif op_rec is not None:
                # TODO: support fixed values with kwargs as well.
                constant_op = np.array(op_rec)
                in_graph_fn_column.op_records[i].op = constant_op
                in_graph_fn_column.op_records[i].space = get_space_from_op(constant_op)
                self.constant_op_records.add(in_graph_fn_column.op_records[i])

        if len(out_graph_fn_column.op_records) == 1:
            return out_graph_fn_column.op_records[0]
        else:
            return tuple(out_graph_fn_column.op_records)

    def call_api(self, method, method_owner, *params, **kwargs):
        """
        Executes an assembly run through another API method (will actually call this API method for further assembly).

        Args:
            method (callable): The method (graph_fn or API method) to call.
            method_owner (Component): Component this method belongs to.
            *params (Union[DataOpRecord,np.array,numeric]): The DataOpRecords (or constant values) to be used for
                calling the method.

        Returns:
            DataOpRecord: Output op of calling this api method.
        """
        api_method_rec = method_owner.api_methods[method.__name__]

        params_no_none = list()
        for p in params[::-1]:
            # Only allow Nones at end of params (positional default args).
            if p is None:
                assert len(params_no_none) == 0,\
                    "ERROR: params ({}) to API-method '{}' have Nones amongst them (ok, if at the end of the params " \
                    "list, but not in the middle).".format(params, method.__name__)
            else:
                params_no_none.insert(0, p)
        # Add all kwargs as tuples (key, op-rec) to params_no_none.
        for key, value in kwargs.items():
            params_no_none.append(tuple([key, value]))

        # Create op-record column to call API method with. Ignore None input params. These should not be sent
        # to the API-method.
        in_op_column = DataOpRecordColumnIntoAPIMethod(
            op_records=len(params_no_none), component=self, api_method_rec=api_method_rec
        )

        # Add the column to the API-method record.
        api_method_rec.in_op_columns.append(in_op_column)

        # Link from incoming op_recs into the new column or populate new column with ops/Spaces (this happens
        # if this call was made from within a graph_fn such that ops and Spaces are already known).
        flex = None
        for i, op_rec in enumerate(params_no_none):
            # Named arg/kwarg -> get input_name from that and peel op_rec.
            if isinstance(op_rec, tuple) and not isinstance(op_rec, DataOpTuple):
                key = op_rec[0]
                op_rec = op_rec[1]
                in_op_column.op_records[i].kwarg = key
            # Positional arg -> get input_name from input_names list.
            else:
                key = api_method_rec.input_names[i if flex is None else flex]

            # Var-positional arg, attach the actual position to input_name string.
            if method_owner.api_method_inputs[key] == "*flex":
                if flex is None:
                    flex = i
                key += "[{}]".format(i - flex)

            # We are already in building phase (params may be coming from inside graph_fn).
            if self.graph_builder is not None and self.graph_builder.phase == "building":
                # Params are op-records -> link AND pass on actual ops/Spaces.
                if isinstance(op_rec, DataOpRecord):
                    in_op_column.op_records[i].op = op_rec.op
                    in_op_column.op_records[i].space = method_owner.api_method_inputs[key] = \
                        get_space_from_op(op_rec.op)
                    op_rec.next.add(in_op_column.op_records[i])
                    in_op_column.op_records[i].previous = op_rec
                # Params are actual ops: Pass them (and Space) on to in-column records.
                else:
                    in_op_column.op_records[i].op = op_rec
                    in_op_column.op_records[i].space = method_owner.api_method_inputs[key] = \
                        get_space_from_op(op_rec)
                # Check input-completeness of Component (but not strict as we are only calling API, not a graph_fn).
                if method_owner.input_complete is False:
                    # Check Spaces and create variables.
                    method_owner.graph_builder.build_component_when_input_complete(method_owner)

            # A DataOpRecord from the meta-graph.
            elif isinstance(op_rec, DataOpRecord):
                op_rec.next.add(in_op_column.op_records[i])
                in_op_column.op_records[i].previous = op_rec
                if key not in method_owner.api_method_inputs:
                    method_owner.api_method_inputs[key] = None
            # Fixed value (instead of op-record): Store the fixed value directly in the op.
            else:
                in_op_column.op_records[i].op = np.array(op_rec)
                in_op_column.op_records[i].space = get_space_from_op(op_rec)
                if key not in method_owner.api_method_inputs or method_owner.api_method_inputs[key] is None:
                    method_owner.api_method_inputs[key] = in_op_column.op_records[i].space
                self.constant_op_records.add(in_op_column.op_records[i])

        # Now actually call the API method with that the correct *args and **kwargs from the new column and
        # create a new out-column with num-records == num-return values.
        name = method.__name__
        self.logger.debug("Calling api method {} with owner {}:".format(name, method_owner))
        args = [op_rec for op_rec in in_op_column.op_records if op_rec.kwarg is None]
        kwargs = {op_rec.kwarg: op_rec for op_rec in in_op_column.op_records if op_rec.kwarg is not None}
        out_op_recs = method(*args, **kwargs)

        # Process the results (push into a column).
        out_op_recs = util.force_list(out_op_recs)
        out_op_column = DataOpRecordColumnFromAPIMethod(
            op_records=len(out_op_recs),
            component=self,
            api_method_name=method.__name__
        )

        # Link the returned ops to that new out-column.
        for i, op_rec in enumerate(out_op_recs):
            # If we already have actual op(s) and Space(s), push them already into the
            # DataOpRecordColumnFromAPIMethod's records.
            if self.graph_builder is not None and self.graph_builder.phase == "building":
                out_op_column.op_records[i].op = op_rec.op
                out_op_column.op_records[i].space = op_rec.space
            op_rec.next.add(out_op_column.op_records[i])
            out_op_column.op_records[i].previous = op_rec
        # And append the new out-column to the api-method-rec.
        api_method_rec.out_op_columns.append(out_op_column)

        # Then return the op-records from the new out-column.
        if len(out_op_column.op_records) == 1:
            return out_op_column.op_records[0]
        else:
            return tuple(out_op_column.op_records)

    def sanity_check_call_parameters(self, params, method, method_type, add_auto_key_as_first_param):
        raw_signature_parameters = inspect.signature(method).parameters
        actual_params = list(raw_signature_parameters.values())
        if add_auto_key_as_first_param is True:
            actual_params = actual_params[1:]
        if len(params) != len(actual_params):
            # Check whether the last arg is var_positional (e.g. *inputs; in that case it's ok if the number of params
            # is larger than that of the actual graph_fn params or its one smaller).
            if actual_params[-1].kind == inspect.Parameter.VAR_POSITIONAL and (len(params) > len(actual_params) > 0 or
                                                                               len(params) == len(actual_params) - 1):
                pass
            # Some actual params have default values: Number of given params must be at least as large as the number
            # of non-default actual params but maximally as large as the number of actual_parameters.
            elif len(actual_params) >= len(params) >= sum(
                    [p.default is inspect.Parameter.empty for p in actual_params]):
                pass
            else:
                raise RLGraphError(
                    "ERROR: {} '{}/{}' has {} input-parameters, but {} ({}) were being provided in the "
                    "`Component.call` method!".format(method_type, self.name, method.__name__,
                                                      len(actual_params), len(params), params)
                )

    def get_number_of_allowed_inputs(self, api_method_name):
        """
        Returns the number of allowed input args for a given API-method.

        Args:
            api_method_name (str): The API-method to analyze.

        Returns:
            Tuple[int,int]: A tuple with the range (lower/upper bound) of allowed input args for the given API-method.
                An upper bound of None means that the API-method accepts any number of input args equal or larger
                than the lower bound.
        """
        input_names = self.api_methods[api_method_name].input_names
        num_allowed_inputs = [0, 0]
        for in_name in input_names:
            # Positional arg with default values (not required, only raise upper bound).
            if self.api_method_inputs[in_name] == "flex":
                num_allowed_inputs[1] += 1
            # Var-positional (no upper bound anymore).
            elif self.api_method_inputs[in_name] == "*flex":
                num_allowed_inputs[1] = None
            # Required arg (raise both lower and upper bound).
            else:
                num_allowed_inputs[0] += 1
                num_allowed_inputs[1] += 1

        return tuple(num_allowed_inputs)

    def check_input_completeness(self):
        """
        Checks whether this Component is "input-complete" and stores the result in self.input_complete.
        Input-completeness is reached (only once and then it stays that way) if all API-methods of this component
        (whose `must_be_complete` field is not set to False) have all their input Spaces defined.

        Returns:
            bool: Whether this Component is input_complete or not.
        """
        assert self.input_complete is False

        self.input_complete = True
        # Loop through all API methods.
        for method_name, api_method_rec in self.api_methods.items():
            # This API method doesn't have to be completed, ignore and don't add it to space_dict.
            if api_method_rec.must_be_complete is False:
                continue

            # Loop through all of this API-method's input parameter names and check, whether they
            # all have a Space  defined.
            for input_name in api_method_rec.input_names:
                assert input_name in self.api_method_inputs
                # This one is not defined yet -> Component is not input-complete.
                if self.api_method_inputs[input_name] is None:
                    self.input_complete = False
                    return False
                # OR: API-method has a var-positional parameter: Check whether it has been called at least once (in
                # which case we have Space information stored under "*args[0]").
                elif self.api_method_inputs[input_name] == "*flex":
                    # Check all keys "input_name[n]" for any None. If one None found -> input incomplete.
                    idx = 0
                    while True:
                        key = input_name+"["+str(idx)+"]"
                        if key not in self.api_method_inputs:
                            # We require at least one param if the flex param is the only param. Otherwise, none are ok.
                            if idx > (0 if len(api_method_rec.input_names) == 1 else -1):
                                break
                            # No input defined (has not been called) -> Not input complete.
                            else:
                                self.input_complete = False
                                return False
                        elif self.api_method_inputs[key] is None:
                            self.input_complete = False
                            return False
                        idx += 1
        return True

    def check_variable_completeness(self):
        """
        Checks, whether this Component is input-complete AND all our sub-Components are input-complete.
        At that point, all variables are defined and we can run the `_variables` graph_fn.

        Returns:
            bool: Whether this Component is "variables-complete".
        """
        # We are already variable-complete -> shortcut return here.
        if self.variable_complete:
            return True
        # We are not input-complete yet (our own variables have not been created) -> return False.
        elif self.input_complete is False:
            return False

        # Simply check all sub-Components for input-completeness.
        self.variable_complete = all(sc.input_complete for sc in self.sub_components.values())
        return self.variable_complete

    def when_input_complete(self, input_spaces=None, action_space=None, device=None, summary_regexp=None):
        """
        Wrapper that calls both `self.check_input_spaces` and `self.create_variables` in sequence and passes
        the dict with the input_spaces for each argument (key=arg name) and the action_space as parameter.

        Args:
            input_spaces (Optional[Dict[str,Space]]): A dict with Space/shape information.
                keys=in-argument name (str); values=the associated Space.
                Use None to take `self.api_method_inputs` instead.
            action_space (Optional[Space]): The action Space of the Agent/GraphBuilder. Can be used to construct and connect
                more Components (which rely on this information). This eliminates the need to pass the action Space
                information into many Components' constructors.
            device (str): The device to use for the variables generated.
            summary_regexp (Optional[str]): A regexp (str) that defines, which summaries should be generated
                and registered.
        """
        # Store the summary_regexp to use.
        self.summary_regexp = summary_regexp

        input_spaces = input_spaces or self.api_method_inputs

        # Allow the Component to check its input Space.
        self.check_input_spaces(input_spaces, action_space)
        # Allow the Component to create all its variables.
        if get_backend() == "tf":
            with tf.device(device):
                if self.reuse_variable_scope:
                    with tf.variable_scope(name_or_scope=self.reuse_variable_scope, reuse=tf.AUTO_REUSE):
                        self.create_variables(input_spaces, action_space)
                else:
                    with tf.variable_scope(self.global_scope):
                        self.create_variables(input_spaces, action_space)
        elif get_backend() == "pytorch":
            # No scoping/devices here, handled at tensor level.
            self.create_variables(input_spaces, action_space)
        # Add all created variables up the parent/container hierarchy.
        self.propagate_variables()

    def check_input_spaces(self, input_spaces, action_space=None):
        """
        Should check on the nature of all in-Sockets Spaces of this Component. This method is called automatically
        by the Model when all these Spaces are know during the Model's build time.

        Args:
            input_spaces (Dict[str,Space]): A dict with Space/shape information.
                keys=in-Socket name (str); values=the associated Space
            action_space (Optional[Space]): The action Space of the Agent/GraphBuilder. Can be used to construct and
                connect more Components (which rely on this information). This eliminates the need to pass the
                action Space information into many Components' constructors.
        """
        pass

    def create_variables(self, input_spaces, action_space=None):
        """
        Should create all variables that are needed within this component,
        unless a variable is only needed inside a single _graph_fn-method, in which case,
        it should be created there.
        Variables must be created via the backend-agnostic self.get_variable-method.

        Note that for different scopes in which this component is being used, variables will not(!) be shared.

        Args:
            input_spaces (Dict[str,Space]): A dict with Space/shape information.
                keys=in-Socket name (str); values=the associated Space
            action_space (Optional[Space]): The action Space of the Agent/GraphBuilder. Can be used to construct and
                connect more Components (which rely on this information). This eliminates the need to pass the action
                Space information into many Components' constructors.
        """
        pass

    def register_variables(self, *variables):
        """
        Adds already created Variables to our registry. This could be useful if the variables are not created
        by our own `self.get_variable` method, but by some backend-specific object (e.g. tf.layers).
        Also auto-creates summaries (regulated by `self.summary_regexp`) for the given variables.

        Args:
            variables (SingleDataOp): The Variable objects to register.
        """
        for var in variables:
            # Use our global_scope plus the var's name without anything in between.
            # e.g. var.name = "dense-layer/dense/kernel:0" -> key = "dense-layer/kernel"
            # key = re.sub(r'({}).*?([\w\-.]+):\d+$'.format(self.global_scope), r'\1/\2', var.name)
            key = re.sub(r':\d+$', "", var.name)
            self.variables[key] = var

            # Auto-create the summary for the variable.
            summary_name = var.name[len(self.global_scope) + (1 if self.global_scope else 0):]
            summary_name = re.sub(r':\d+$', "", summary_name)
            self.create_summary(summary_name, var)

    def get_variable(self, name="", shape=None, dtype="float", initializer=None, trainable=True,
                     from_space=None, add_batch_rank=False, add_time_rank=False, time_major=False, flatten=False):
        """
        Generates or returns a variable to use in the selected backend.
        The generated variable is automatically registered in this component's (and all parent components')
        variable-registry under its global-scoped name.

        Args:
            name (str): The name under which the variable is registered in this component.
            shape (Optional[tuple]): The shape of the variable. Default: empty tuple.
            dtype (Union[str,type]): The dtype (as string) of this variable.
            initializer (Optional[any]): Initializer for this variable.
            trainable (bool): Whether this variable should be trainable. This will be overwritten,
                if the Component has its own `trainable` property set to either True or False.
            from_space (Optional[Space,str]): Whether to create this variable from a Space object
                (shape and dtype are not needed then). The Space object can be given directly or via the name
                of the in-Socket holding the Space.
            add_batch_rank (Optional[bool,int]): If True and `from_space` is given, will add a 0th (1st) rank (None) to
                the created variable. If it is an int, will add that int instead of None.
                Default: False.
            add_time_rank (Optional[bool,int]): If True and `from_space` is given, will add a 1st (0th) rank (None) to
                the created variable. If it is an int, will add that int instead of None.
                Default: False.
            time_major (bool): Only relevant if both `add_batch_rank` and `add_time_rank` are True.
                Will make the time-rank the 0th rank and the batch-rank the 1st rank.
                Otherwise, batch-rank will be 0th and time-rank will be 1st.
                Default: False.
            flatten (bool): Whether to produce a FlattenedDataOp with auto-keys.

        Returns:
            DataOp: The actual variable (dependent on the backend) or - if from
                a ContainerSpace - a FlattenedDataOp or ContainerDataOp depending on the Space.
        """

        # Overwrite the given trainable parameter, iff self.trainable is actually defined as a bool.
        trainable = self.trainable if self.trainable is not None else (trainable if trainable is not None else True)

        # Called as getter.
        if shape is None and initializer is None and from_space is None:
            if name not in self.variables:
                raise KeyError(
                    "Variable with name '{}' not found in registry of Component '{}'!".format(name, self.name)
                )
            # TODO: Maybe try both the pure name AND the name with global-scope in front.
            return self.variables[name]

        # Called as setter.
        var = None

        # We are creating the variable using a Space as template.
        if from_space is not None:
            var = self._variable_from_space(
                flatten, from_space, name, add_batch_rank, add_time_rank, time_major, trainable, initializer
            )

        # TODO: Figure out complete concept for python/numpy based Components (including their handling of variables).
        # Assume that when using pytorch, we use Python/numpy collections to store data.
        elif self.backend == "python" or get_backend() == "python" or get_backend() == "pytorch":
            if add_batch_rank is not False and isinstance(add_batch_rank, int):
                if isinstance(add_time_rank, int):
                    if time_major:
                        var = [[initializer for _ in range_(add_batch_rank)] for _ in range_(add_time_rank)]
                    else:
                        var = [[initializer for _ in range_(add_time_rank)] for _ in range_(add_batch_rank)]
                else:
                    var = [initializer for _ in range_(add_batch_rank)]
            elif add_time_rank is not False and isinstance(add_time_rank, int):
                var = [initializer for _ in range_(add_time_rank)]
            elif initializer is not None:
                # Return
                var = initializer
            else:
                var = []
            return var

        # Direct variable creation (using the backend).
        elif get_backend() == "tf":
            # Provide a shape, if initializer is not given or it is an actual Initializer object (rather than an array
            # of fixed values, for which we then don't need a shape as it comes with one).
            if initializer is None or isinstance(initializer, tf.keras.initializers.Initializer):
                shape = tuple((() if add_batch_rank is False else
                               (None,) if add_batch_rank is True else (add_batch_rank,)) + (shape or ()))
            # Numpyize initializer and give it correct dtype.
            else:
                shape = None
                initializer = np.asarray(initializer, dtype=util.dtype(dtype, "np"))

            var = tf.get_variable(
                name=name, shape=shape, dtype=util.dtype(dtype), initializer=initializer, trainable=trainable
            )
        elif get_backend() == "tf-eager":
            shape = tuple(
                (() if add_batch_rank is False else (None,) if add_batch_rank is True else (add_batch_rank,)) +
                (shape or ())
            )

            var = eager.Variable(
                name=name, shape=shape, dtype=util.dtype(dtype), initializer=initializer, trainable=trainable
            )

        # TODO: what about python variables?
        # Registers the new variable with this Component.
        key = ((self.reuse_variable_scope + "/") if self.reuse_variable_scope else
               (self.global_scope + "/") if self.global_scope else "") + name
        # Container-var: Save individual Variables.
        # TODO: What about a var from Tuple space?
        if isinstance(var, OrderedDict):
            for sub_key, v in var.items():
                self.variables[key + sub_key] = v
        else:
            self.variables[key] = var

        return var

    def _variable_from_space(self, flatten, from_space, name, add_batch_rank, add_time_rank, time_major, trainable,
                             initializer):
        """
        Private variable from space helper, see 'get_variable' for API.
        """
        # Variables should be returned in a flattened OrderedDict.
        if get_backend() == "tf":
            if self.reuse_variable_scope is not None:
                with tf.variable_scope(name_or_scope=self.reuse_variable_scope, reuse=True):
                    if flatten:
                        return from_space.flatten(mapping=lambda key_, primitive: primitive.get_variable(
                            name=name + key_, add_batch_rank=add_batch_rank, add_time_rank=add_time_rank,
                            time_major=time_major, trainable=trainable, initializer=initializer,
                            is_python=(self.backend == "python" or get_backend() == "python")
                        ))
                    # Normal, nested Variables from a Space (container or primitive).
                    else:
                        return from_space.get_variable(
                            name=name, add_batch_rank=add_batch_rank, trainable=trainable, initializer=initializer,
                            is_python=(self.backend == "python" or get_backend() == "python")
                        )
            else:
                if flatten:
                    return from_space.flatten(mapping=lambda key_, primitive: primitive.get_variable(
                        name=name + key_, add_batch_rank=add_batch_rank, add_time_rank=add_time_rank,
                        time_major=time_major, trainable=trainable, initializer=initializer,
                        is_python=(self.backend == "python" or get_backend() == "python")
                    ))
                # Normal, nested Variables from a Space (container or primitive).
                else:
                    return from_space.get_variable(
                        name=name, add_batch_rank=add_batch_rank, trainable=trainable, initializer=initializer,
                        is_python=(self.backend == "python" or get_backend() == "python")
                    )

    def get_variables(self, *names, **kwargs):
        """
        Utility method to get one or more component variable(s) by name(s).

        Args:
            names (List[str]): Lookup name strings for variables. None for all.

        Keyword Args:
            collections (set): A set of collections to which the variables have to belong in order to be returned here.
                Default: tf.GraphKeys.TRAINABLE_VARIABLES
            custom_scope_separator (str): The separator to use in the returned dict for scopes.
                Default: '/'.
            global_scope (bool): Whether to use keys in the returned dict that include the global-scopes of the
                Variables. Default: False.

        Returns:
            dict: A dict mapping variable names to their get_backend variables.
        """
        if get_backend() == "tf":
            collections = kwargs.pop("collections", None) or tf.GraphKeys.GLOBAL_VARIABLES
            custom_scope_separator = kwargs.pop("custom_scope_separator", "/")
            global_scope = kwargs.pop("global_scope", True)
            assert not kwargs, "{}".format(kwargs)

            if len(names) == 1 and isinstance(names[0], list):
                names = names[0]
            names = util.force_list(names)
            # Return all variables of this Component (for some collection).
            if len(names) == 0:
                collection_variables = tf.get_collection(collections)
                ret = dict()
                for v in collection_variables:
                    lookup = re.sub(r':\d+$', "", v.name)
                    if lookup in self.variables:
                        if global_scope:
                            # Replace the scope separator with a custom one.
                            ret[re.sub(r'(/|{}|{})'.format(FLAT_TUPLE_CLOSE, FLAT_TUPLE_OPEN),
                                       custom_scope_separator, lookup)] = v
                        else:
                            ret[re.sub(r'^.+/', "", lookup)] = v
                return ret
            # Return only variables of this Component by name.
            else:
                return self.get_variables_by_name(
                    *names, custom_scope_separator=custom_scope_separator, global_scope=global_scope
                )
        elif get_backend() == "pytorch":
            # Just return variables for this component.
            custom_scope_separator = kwargs.pop("custom_scope_separator", "/")
            global_scope = kwargs.pop("global_scope", True)
            return self.get_variables_by_name(
                *names, custom_scope_separator=custom_scope_separator, global_scope=global_scope
            )

    def get_variables_by_name(self, *names, **kwargs):
        """
        Retrieves this components variables by name.

        Args:
            names (List[str]): List of names of Variable to return.

        Keyword Args:
            custom_scope_separator (str): The separator to use in the returned dict for scopes.
                Default: '/'.
            global_scope (bool): Whether to use keys in the returned dict that include the global-scopes of the
                Variables. Default: False.

        Returns:
            dict: Dict containing the requested names as keys and variables as values.
        """
        custom_scope_separator = kwargs.pop("custom_scope_separator", "/")
        global_scope = kwargs.pop("global_scope", False)
        assert not kwargs

        variables = dict()
        for name in names:
            global_scope_name = ((self.global_scope + "/") if self.global_scope else "") + name
            if name in self.variables:
                variables[re.sub(r'/', custom_scope_separator, name)] = self.variables[name]
            elif global_scope_name in self.variables:
                if global_scope:
                    variables[re.sub(r'/', custom_scope_separator, global_scope_name)] = self.variables[
                        global_scope_name]
                else:
                    variables[name] = self.variables[global_scope_name]
        return variables

    def create_summary(self, name, values, type_="histogram"):
        """
        Creates a summary op (and adds it to the graph).
        Skips those, whose full name does not match `self.summary_regexp`.

        Args:
            name (str): The name for the summary. This has to match `self.summary_regexp`.
                The name should not contain a "summary"-prefix or any global scope information
                (both will be added automatically by this method).
            values (op): The op to summarize.
            type_ (str): The summary type to create. Currently supported are:
                "histogram", "scalar" and "text".
        """
        # Prepend the "summaries/"-prefix.
        name = "summaries/" + name
        # Get global name.
        global_name = ((self.global_scope + "/") if self.global_scope else "") + name
        # Skip non matching summaries (all if summary_regexp is None).
        if self.summary_regexp is None or not re.search(self.summary_regexp, global_name):
            return

        summary = None
        if get_backend() == "tf":
            ctor = getattr(tf.summary, type_)
            summary = ctor(name, values)

        # Registers the new summary with this Component.
        if global_name in self.summaries:
            raise RLGraphError("ERROR: Summary with name '{}' already exists in {}'s summary "
                            "registry!".format(global_name, self.name))
        self.summaries[global_name] = summary
        self.propagate_summary(global_name)

    def propagate_summary(self, key_):
        """
        Propagates a single summary op of this Component to its parents' summaries registries.

        Args:
            key_ (str): The lookup key for the summary to propagate.
        """
        # Return if there is no parent.
        if self.parent_component is None:
            return

        # If already there -> Error.
        if key_ in self.parent_component.summaries:
            raise RLGraphError("ERROR: Summary registry of '{}' already has a summary under key '{}'!".
                format(self.parent_component.name, key_))
        self.parent_component.summaries[key_] = self.summaries[key_]

        # Recurse up the container hierarchy.
        self.parent_component.propagate_summary(key_)

    def define_api_method(self, name, func=None, must_be_complete=True, ok_to_overwrite=False, **kwargs):
        """
        Creates a very basic graph_fn based API method for this Component.
        Alternative way for defining an API method directly in the Component via def.

        Args:
            name (Union[str,callable]): The name of the API method to create or one of the Component's method to
                create the API-method from.
            func (Optional[callable]): The graph_fn to wrap or the custom function to set as API-method.
            must_be_complete (bool): Whether this API-method must have all its incoming Spaces defined in order
                for the Component to count as "input-complete". Some API-methods may be still input-incomplete without
                affecting the Component's build process.
            ok_to_overwrite (bool): Whether to raise an Error if an API-method with that name or another property
                with that name already exists. Default: False.

        Keyword Args:
            flatten_ops (bool,Set[int]): See `self.call` for details.
            split_ops (bool,Set[int]): See `self.call` for details.
            add_auto_key_as_first_param (bool): See `self.call` for details.
        """
        # Raise errors if `name` already taken in this Component.
        if not ok_to_overwrite:
            # There already is an API-method with that name.
            if name in self.api_methods:
                raise RLGraphError("API-method with name '{}' already defined!".format(name))
            # There already is another object property with that name (avoid accidental overriding).
            elif getattr(self, name, None) is not None:
                raise RLGraphError("Component '{}' already has a property called '{}'."
                                   " Cannot define an API-method with "
                                   "the same name!".format(self.name, name))

        # Do not build this API as per ctor instructions.
        if name in self.switched_off_apis:
            return

        func_type = util.get_method_type(func)

        # Function is a graph_fn: Build a simple wrapper API-method around it and name it `name`.
        if func_type == "graph_fn":
            def api_method(self, *inputs_, **kwargs_):
                # Mix in user provided kwargs with the setting ones from the original `define_api_method` call.
                default_dict(kwargs_, kwargs)
                func_ = getattr(self, func.__name__)

                return self.call(func_, *inputs_, **kwargs_)

        # Function is a (custom) API-method. Register it with this Component.
        else:
            api_method = func

        self.synthetic_methods.add(name)
        setattr(self, name, api_method.__get__(self, self.__class__))
        setattr(api_method, "__self__", self)
        setattr(api_method, "__name__", name)

        self.api_methods[name] = APIMethodRecord(
            getattr(self, name), component=self, must_be_complete=must_be_complete,
            is_graph_fn_wrapper=(func_type == "graph_fn"),
            add_auto_key_as_first_param=kwargs.get("add_auto_key_as_first_param", False)
        )

        # Direct callable for eager/define by run.
        self.api_fn_by_name[name] = api_method

        # Update the api_method_inputs dict (with empty Spaces if not defined yet).
        # Note: Skip first param of graph_func's input param list if add-auto-key option is True (1st param would be
        # the auto-key then). Also skip if api_method is an unbound function (then 1st param is usually `self`).
        if (func_type == "graph_fn" and kwargs.get("add_auto_key_as_first_param") is True) or \
                (func_type != "graph_fn" and type(api_method).__name__ == "function"):
            skip_1st_arg = 1
        else:
            skip_1st_arg = 0
        param_list = list(inspect.signature(func).parameters.values())[skip_1st_arg:]
        self._complement_api_method_registries(name, param_list)

    def _complement_api_method_registries(self, api_method_name, param_list):
        self.api_methods[api_method_name].input_names = list()
        for param in param_list:
            self.api_methods[api_method_name].input_names.append(param.name)
            if param.name not in self.api_method_inputs:
                # This param has a default value.
                if param.default != inspect.Parameter.empty:
                    # Default is None. Set to "flex" (to signal that this Space is not needed for input-completeness)
                    # and wait for first call using this parameter (only then set it to that Space).
                    if param.default is None:
                        self.api_method_inputs[param.name] = "flex"
                    # Default is some python value (e.g. a bool). Use that are the assigned Space.
                    else:
                        space = get_space_from_op(param.default)
                        self.api_method_inputs[param.name] = space
                # This param is an *args param. Store are "*flex". Then with upcoming API calls, we determine the Spaces
                # for the single items in *args and set them under "param[0]", "param[1]", etc..
                elif param.kind == inspect.Parameter.VAR_POSITIONAL:
                    self.api_method_inputs[param.name] = "*flex"
                # Normal POSITIONAL_ONLY parameter. Store as None (needed) for now.
                else:
                    self.api_method_inputs[param.name] = None

    def add_components(self, *components, **kwargs):
        """
        Adds sub-components to this one.

        Args:
            components (List[Component]): The list of Component objects to be added into this one.

        Keyword Args:
            expose_apis (Optional[Set[str]]): An optional set of strings with API-methods of the child component
                that should be exposed as the parent's API via a simple wrapper API-method for the parent (that
                calls the child's API-method).
        """
        expose_apis = kwargs.pop("expose_apis", set())
        if isinstance(expose_apis, str):
            expose_apis = {expose_apis}

        for component in components:
            # Try to create Component from spec.
            if not isinstance(component, Component):
                component = Component.from_spec(component)
            # Make sure no two components with the same name are added to this one (own scope doesn't matter).
            if component.name in self.sub_components:
                raise RLGraphError("ERROR: Sub-Component with name '{}' already exists in this one!".
                                   format(component.name))
            # Make sure each Component can only be added once to a parent/container Component.
            elif component.parent_component is not None:
                raise RLGraphError("ERROR: Sub-Component with name '{}' has already been added once to a container "
                                "Component! Each Component can only be added once to a parent.".format(component.name))
            # Make sure we don't add to ourselves.
            elif component is self:
                raise RLGraphError("ERROR: Cannot add a Component ({}) as a sub-Component to itself!".format(self.name))
            component.parent_component = self
            self.sub_components[component.name] = component

            # Add own reusable scope to front of sub-components'.
            if self.reuse_variable_scope is not None:
                # Propagate reuse_variable_scope down to the added Component's sub-components.
                component.propagate_subcomponent_properties(
                    properties=dict(reuse_variable_scope=self.reuse_variable_scope)
                )
            # Fix the sub-component's (and sub-sub-component's etc..) scope(s).
            self.propagate_scope(component)

            # Execution modes must be coherent within one component subgraph.
            self.propagate_subcomponent_properties(
                properties=dict(execution_mode=component.execution_mode), component=component
            )

            # Should we expose some API-methods of the child?
            for api_method_name, api_method_rec in component.api_methods.items():
                if api_method_name in expose_apis:
                    def exposed_api_method_wrapper(self, *inputs):
                        return self.call(api_method_rec.method, *inputs)
                    self.define_api_method(api_method_name, exposed_api_method_wrapper)
                    # Add the sub-component's API-registered methods to ours.
                    #self.defined_externally.add(component.scope + "-" + api_method_name)

    def get_all_sub_components(self, list_=None, level_=0):
        """
        Returns all sub-Components (including self) sorted by their nesting-level (... grand-children before children
        before parents).

        Args:
            list_ (Optional[List[Component]])): A list of already collected components to append to.
            level_ (int): The slot indicating the Component level depth in `list_` at which we are currently.

        Returns:
            List[Component]: A list with all the sub-components in `self` and `self` itself.
        """
        return_ = False
        if list_ is None:
            list_ = dict()
            return_ = True
        if level_ not in list_:
            list_[level_] = list()
        list_[level_].append(self)
        level_ += 1
        for sub_component in self.sub_components.values():
            sub_component.get_all_sub_components(list_, level_)
        if return_:
            ret = list()
            for l in sorted(list_.keys(), reverse=True):
                ret.extend(sorted(list_[l], key=lambda c: c.scope))
            return ret

    def get_sub_component_by_scope(self, scope):
        """
        Returns a sub-Component (or None if not found) by scope. The sub-coponent's scope should be given
        as global scope of the sub-component (not local scope with respect to this Component).

        Args:
            scope (str): The global scope of the sub-Component we are looking for.

        Returns:
            Component: The sub-Component with the given global scope if found, None if not found.
        """
        # TODO: make method more efficient.
        components = self.get_all_sub_components()
        for component in components:
            if component.global_scope == scope:
                return component
        return None

    def remove_sub_component_by_name(self, name):
        """
        Removes a sub-component by its name. Raises an error if the sub-component does not exist.

        Args:
            name (str):
        """
        assert name in self.sub_components, "ERROR: Component {} cannot be removed because it is not" \
            "a sub-component. Sub-components by name are: {}.".format(name, list(self.sub_components.keys()))
        self.sub_components.pop(name)

    def get_parents(self):
        """
        Returns a list of parent and grand-parents of this component.

        Returns:
            List[Component]: A list (may be empty if this component has no parents) of all parent and grand-parents.
        """
        ret = list()
        component = self
        while component.parent_component is not None:
            ret.append(component.parent_component)
            component = component.parent_component
        return ret

    def propagate_scope(self, sub_component, properties=None):
        """
        Fixes all the sub-Component's (and its sub-Component's) global_scopes.

        Args:
            sub_component (Optional[Component]): The sub-Component object whose global_scope needs to be updated.
                Use None for this Component itself.
            properties (Optional[dict]): Dict with properties to update on subcomponents.
        """
        # TODO this should be moved to use generic method below, but checking if global scope if set
        # TODO does not work well within that.
        if sub_component is None:
            sub_component = self
        elif self.global_scope:
            sub_component.global_scope = self.global_scope + (
                ("/" + sub_component.scope) if sub_component.scope else "")

        # Recurse.
        for sc in sub_component.sub_components.values():
            sub_component.propagate_scope(sc)

    def propagate_subcomponent_properties(self, properties, component=None):
        """
        Recursively updates properties of component and its sub-components.
        Args:
            properties (dict): Dict with names of properties and their values to recursively update
                sub-components with.
            component (Optional([Component])): Component to recursively update. Uses self if None.
        """
        if component is None:
            component = self
        properties_scoped = copy.deepcopy(properties)
        for name, value in properties.items():
            # Property is some scope (value is then that scope of the parent component).
            # Have to align it with sub-component's local scope.
            if value and (name == "global_scope" or name == "reuse_variable_scope"):
                value += (("/" + component.scope) if component.scope else "")
                properties_scoped[name] = value
                setattr(component, name, value)
            # Normal property: Set to static given value.
            else:
                setattr(component, name, value)
        for sc in component.sub_components.values():
            component.propagate_subcomponent_properties(properties_scoped, sc)

    def propagate_variables(self, keys=None):
        """
        Propagates all variable from this Component to its parents' variable registries.

        Args:
            keys (Optional[List[str]]): An optional list of variable names to propagate. Should only be used in
                internal, recursive calls to this same method.
        """
        # Return if there is no parent.
        if self.parent_component is None:
            return

        # Add all our variables to parent's variable registry.
        keys = keys or self.variables.keys()
        for key in keys:
            # If already there (bubbled up from some child component that was input complete before us)
            # -> Make sure the variable object is identical.
            if key in self.parent_component.variables:
                if self.variables[key] is not self.parent_component.variables[key]:
                    raise RLGraphError("ERROR: Variable registry of '{}' already has a variable under key '{}'!". \
                          format(self.parent_component.name, key))
            self.parent_component.variables[key] = self.variables[key]

        # Recurse up the container hierarchy.
        self.parent_component.propagate_variables(keys)

    def copy(self, name=None, scope=None, device=None, trainable=None, #global_component=False,
             reuse_variable_scope=None):
        """
        Copies this component and returns a new component with possibly another name and another scope.
        The new component has its own variables (they are not shared with the variables of this component as they
        will be created after this copy anyway, during the build phase).
        and is initially not connected to any other component.

        Args:
            name (str): The name of the new Component. If None, use the value of scope.
            scope (str): The scope of the new Component. If None, use the same scope as this component.
            device (str): The device of the new Component. If None, use the same device as this one.
            trainable (Optional[bool]): Whether to make all variables in this component trainable or not.
                Use None for no specific preference.
            #global_component (Optional[bool]): Whether the new Component is global or not. If None, use the same
            #    setting as this one.
            reuse_variable_scope (Optional[str]): If not None, variables of the copy will be shared under this scope.
        Returns:
            Component: The copied component object.
        """
        # Make sure we are still in the assembly phase (should not copy actual ops).
        assert self.input_complete is False, "ERROR: Cannot copy a Component ('{}') that has already been built!". \
            format(self.name)

        if scope is None:
            scope = self.scope
        if name is None:
            name = scope
        if device is None:
            device = self.device
        if trainable is None:
            trainable = self.trainable
        #if global_component is None:
        #    global_component = self.global_component

        # Simply deepcopy self and change name and scope.
        new_component = copy.deepcopy(self)
        new_component.name = name
        new_component.scope = scope
        # Change global_scope for the copy and all its sub-components.
        new_component.global_scope = scope
        new_component.propagate_scope(sub_component=None)

        # Propagate reusable scope, device and other trainable.
        new_component.propagate_subcomponent_properties(
            properties=dict(reuse_variable_scope=reuse_variable_scope, device=device, trainable=trainable)
        )
        #new_component.global_component = global_component
        # Erase the parent pointer.
        new_component.parent_component = None

        return new_component

    @staticmethod
    def scatter_update_variable(variable, indices, updates):
        """
        Updates a variable. Optionally returns the operation depending on the backend.

        Args:
            variable (any): Variable to update.
            indices (array): Indices to update.
            updates (any):  Update values.

        Returns:
            Optional[op]: The graph operation representing the update (or None).
        """
        if get_backend() == "tf":
            return tf.scatter_update(ref=variable, indices=indices, updates=updates)

    @staticmethod
    def assign_variable(ref, value):
        """
        Assigns a variable to a value.

        Args:
            ref (any): The variable to assign to.
            value (any): The value to use for the assignment.

        Returns:
            Optional[op]: None or the graph operation representing the assginment.
        """
        if get_backend() == "tf":
            return tf.assign(ref=ref, value=value)

    @staticmethod
    def read_variable(variable, indices=None):
        """
        Reads a variable.

        Args:
            variable (DataOp): The variable whose value to read.
            indices (Optional[np.ndarray,tf.Tensor]): Indices (if any) to fetch from the variable.

        Returns:
            any: Variable values.
        """
        if get_backend() == "tf":
            if indices is not None:
                # Could be redundant, question is if there may be special read operations
                # in other backends, or read from remote variable requiring extra args.
                return tf.gather(params=variable, indices=indices)
            else:
                return variable
        elif get_backend() == "pytorch":
            # This is not a useful call but for interop in some components
            return variable

    def _graph_fn__variables(self):
        """
        Outputs all of this Component's variables in a DataOpDict (API-method "_variables").

        This can be used e.g. to sync this Component's variables into another Component, which owns
        a Synchronizable() as a sub-component. The returns values of this graph_fn are then sent into
        the other Component's API-method `sync` (parameter: "values") for syncing.

        Returns:
            DataOpDict: Dict with keys=variable names and values=variable (SingleDataOp).
        """
        # Must use custom_scope_separator here b/c RLGraph doesn't allow Dict with '/'-chars in the keys.
        # '/' could collide with a FlattenedDataOp's keys and mess up the un-flatten process.
        variables_dict = self.get_variables(custom_scope_separator="-")
        return DataOpDict(variables_dict)

    def sub_component_by_name(self, scope_name):
        """
        Returns a subcomponent of this component by its name.
        Args:
            scope_name (str): Name of the component. This is typically its scope.

        Returns:
            Component: Subcomponent if it exists.

        Raises:
            ValueError: Error if no subcomponent with this name exists.
        """
        if scope_name not in self.sub_components:
            raise RLGraphError("Name {} is not a valid subcomponent name for component {}. Subcomponents "
                               "are: {}".format(scope_name, self.__str__(), self.sub_components.keys()))
        return self.sub_components[scope_name]

    def __str__(self):
        return "{}('{}' api={})".format(type(self).__name__, self.name, str(list(self.api_methods.keys())))
