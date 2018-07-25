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

import copy
import inspect
import numpy as np
import re

from rlgraph import RLGraphError, get_backend, Specifiable
from rlgraph.utils.ops import SingleDataOp, DataOpDict, DataOpRecord, APIMethodRecord, \
    DataOpRecordColumnIntoGraphFn, DataOpRecordColumnFromGraphFn, DataOpRecordColumnIntoAPIMethod, \
    DataOpRecordColumnFromAPIMethod, GraphFnRecord
from rlgraph.utils import util
from rlgraph.spaces.space import Space


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
            global_component (bool): In distributed mode, this flag indicates if the component is part of the
                shared global model or local to the worker. Defaults to False and will be ignored if set to
                True in non-distributed mode.

            # TODO: remove when we have numpy-based Components (then we can do test calls to infer everything automatically)
            graph_fn_num_outputs (dict): A dict specifying which graph_fns have how many return values.
                This can be useful if graph_fns don't clearly have a fixed number of return values and the auto-inferral
                utility function cannot determine the actual number of returned values.

            switched_off_apis (Optional[Set[str]]): Set of API-method names that should NOT be build for this Component.
            backend (str): The custom backend that this Component obliges to. None to use the RLGraph global backend.
                Default: None.
        """
        # Scope if used to create scope hierarchies inside the Graph.
        # self.logger = logging.getLogger(__name__)
        self.scope = kwargs.pop("scope", "")

        assert re.match(r'^[\w\-]*$', self.scope), \
            "ERROR: scope {} does not match scope-pattern! Needs to be \\w or '-'.".format(self.scope)
        # The global scope string defining the exact nested position of this Component in the Graph.
        # e.g. "/core/component1/sub-component-a"
        self.global_scope = self.scope
        # Names of sub-components that exist (parallelly) inside a containing component must be unique.
        self.name = kwargs.pop("name", self.scope)  # if no name given, use scope
        self.device = kwargs.pop("device", None)
        self.trainable = kwargs.pop("trainable", None)
        self.global_component = kwargs.pop("global_component", False)
        self.graph_fn_num_outputs = kwargs.pop("graph_fn_num_outputs", dict())
        self.switched_off_apis = kwargs.pop("switched_off_apis", set())
        self.backend = kwargs.pop("backend", None)

        assert not kwargs, "ERROR: kwargs ({}) still contains items!".format(kwargs)

        # Keep track of whether this Component has already been added to another Component and throw error
        # if this is done twice. Each Component can only be added once to a parent Component.
        self.parent_component = None  # type: Component

        # Dict of sub-components that live inside this one (key=sub-component's scope).
        self.sub_components = OrderedDict()

        # Dicts holding information about which op-record-tuples go via which API methods into this Component
        # and come out of it.
        # keys=API method name; values=APIMethodRecord
        self.api_methods = self.get_api_methods()
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
        self.summary_regexp = r''

        # Now add all sub-Components.
        self.add_components(*sub_components)

        # Define the "_variables" API-method that each Component automatically has.
        self.define_api_method("_variables", self._graph_fn__variables)

    def get_api_methods(self):
        """
        Detects all methods of the Component that should be registered as API-methods for
        this Component.

        Returns:
            Dict[str,APIMethodRecord]: Dict of kay=API-method name (str); values=APIMethodRecord.
        """
        ret = dict()
        # look for all our API methods (those that use the `call` method).
        for member in inspect.getmembers(self):
            name, method = (member[0], member[1])
            if name != "define_api_method" and name != "add_components" and name[0] != "_" and \
                    name not in self.switched_off_apis and util.get_method_type(method) == "api":
                callable_anytime = False  # not util.does_method_call_graph_fns(method)
                ret[name] = APIMethodRecord(method, component=self, callable_anytime=callable_anytime)
        return ret

    def call(self, method, *params, **kwargs):
        """
        Performs either:
        a) An assembly run through another API method (will actually call this API method for further assembly).
        b) A dry run through a graph_fn (without calling it) just generating the empty op-record-columns around the
            graph_fn (incoming and outgoing).

        Args:
            method (callable): The method (graph_fn or API method) to call.
            *params (DataOpRecord): The DataOpRecords to be used  for calling the method.

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
            ok_to_call_own_api (bool): Whether an Error should be suppressed if a Component `call`s an API-method
                of itself. This is usually not allowed due to introducing circular dependencies.
                Default: False.

        Returns:
            Tuple[DataOpRecord]: The returned tuple of DataOpRecords coming from the called API-method or graph_fn.
        """
        # Owner of method:
        method_owner = method.__self__

        ok_to_call_own_api = kwargs.pop("ok_to_call_own_api", False)

        # Method is an API method.
        if method.__name__ in method_owner.api_methods:
            if method_owner is self and ok_to_call_own_api is False:
                parent_caller = inspect.stack()[1][3]
                raise RLGraphError("'{}' Component's API-method ('{}') cannot `call` another API-method ('{}') of the "
                                "same Component!".format(self.name, parent_caller, method.__name__))
            return self.call_api(method, method_owner, *params)

        # Method is a graph_fn.
        else:
            return self.call_graph_fn(method, method_owner, *params, **kwargs)

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
        # Make sure the graph_fn belongs to this Component (not allowed to call graph_fn of other component
        # directly).
        if method_owner is not self:
            raise RLGraphError("Graph_fn '{}' may only be sent to `call` by its owner ({})! However, '{}' is "
                            "calling it.".format(method.__name__, method_owner.scope, self.scope))

        # Sanity check number of actual graph_fn input values against len(params).
        actual_params = list(inspect.signature(method).parameters.values())
        if kwargs.get("add_auto_key_as_first_param") is True:
            actual_params = actual_params[1:]
        if len(params) != len(actual_params):
            # Check whether the last arg is var_positional (e.g. *inputs; in that case it's ok if the number of params
            # is larger than that of the actual graph_fn params).
            if len(params) > len(actual_params) > 0 and actual_params[-1].kind == inspect.Parameter.VAR_POSITIONAL:
                pass
            # Some actual params have default values: Number of given params must be at least as large as the number
            # of non-default actual params but maximally as large as the number of actual_parameters.
            elif len(actual_params) >= len(params) >= sum(
                    [p.default is inspect.Parameter.empty for p in actual_params]):
                pass
            else:
                raise RLGraphError("ERROR: Graph_fn '{}/{}' has {} input-parameters, but {} ({}) were being provided in "
                                "the `Component.call` method!".
                                format(self.name, method.__name__, len(inspect.signature(method).parameters),
                                       len(params), params))

        # Store a  graph_fn record in this component for better in/out-op-record-column reference.
        if method.__name__ not in self.graph_fns:
            self.graph_fns[method.__name__] = GraphFnRecord(graph_fn=method, component=self)

        # Create 2 op-record columns, one going into the graph_fn and one getting out of there and link
        # them together via the graph_fn (w/o calling it).
        # TODO: remove when we have numpy-based Components (then we can do test calls to infer everything automatically)
        if method.__name__ in self.graph_fn_num_outputs:
            num_graph_fn_return_values = self.graph_fn_num_outputs[method.__name__]
        else:
            num_graph_fn_return_values = util.get_num_return_values(method)
        self.logger.debug("Graph_fn has {} return values (inferred).".format(method.__name__,
                                                                             num_graph_fn_return_values))

        # Generate the two op-rec-columns (in-going and out-coming) and link them together.
        in_graph_fn_column = DataOpRecordColumnIntoGraphFn(len(params), component=self, graph_fn=method, **kwargs)
        # If in-column is empty, add it to the "empty in-column" set.
        if len(in_graph_fn_column.op_records) == 0:
            self.no_input_graph_fn_columns.add(in_graph_fn_column)
        self.graph_fns[method.__name__].in_op_columns.append(in_graph_fn_column)

        out_graph_fn_column = DataOpRecordColumnFromGraphFn(
            num_graph_fn_return_values, component=self, graph_fn_name=method.__name__,
            in_graph_fn_column=in_graph_fn_column
        )
        in_graph_fn_column.out_graph_fn_column = out_graph_fn_column
        self.graph_fns[method.__name__].out_op_columns.append(out_graph_fn_column)

        # Link from in_op_recs into the new column (and back).
        for i, op_rec in enumerate(params):
            if not isinstance(op_rec, DataOpRecord):
                in_graph_fn_column.op_records[i].op = np.array(op_rec)
                in_graph_fn_column.op_records[i].space = 0
                self.constant_op_records.add(in_graph_fn_column.op_records[i])
            else:
                op_rec.next.add(in_graph_fn_column.op_records[i])
                in_graph_fn_column.op_records[i].previous = op_rec

        if len(out_graph_fn_column.op_records) == 1:
            return out_graph_fn_column.op_records[0]
        else:
            return out_graph_fn_column.op_records

    def call_api(self, method, method_owner, *params):
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

        # Create op-record column to call API method with. Ignore None input params. These should not be sent
        # to the API-method.
        in_op_column = DataOpRecordColumnIntoAPIMethod(op_records=len(params_no_none),
                                                       component=self,
                                                       api_method_rec=api_method_rec)
        api_method_rec.in_op_columns.append(in_op_column)

        # Link from in_op_recs into the new column.
        for i, op_rec in enumerate(params_no_none):
            # Fixed value (instead of op-record): Store the fixed value directly in the op.
            if not isinstance(op_rec, DataOpRecord):
                in_op_column.op_records[i].op = np.array(op_rec)
                in_op_column.op_records[i].space = 0
                self.constant_op_records.add(in_op_column.op_records[i])
            else:
                op_rec.next.add(in_op_column.op_records[i])
                in_op_column.op_records[i].previous = op_rec

        # Now actually call the API method with that column and
        # create a new out-column with num-records == num-return values.
        name = method.__name__
        self.logger.debug("Calling api method {} with owner {}:".format(name, method_owner))
        out_op_recs = method(*in_op_column.op_records)
        out_op_recs = util.force_list(out_op_recs)
        out_op_column = DataOpRecordColumnFromAPIMethod(
            op_records=len(out_op_recs),
            component=self,
            api_method_name=method.__name__
        )

        # Link the returned ops to that new out-column.
        for i, op_rec in enumerate(out_op_recs):
            op_rec.next.add(out_op_column.op_records[i])
            out_op_column.op_records[i].previous = op_rec
        # And append the new out-column to the api-method-rec.
        api_method_rec.out_op_columns.append(out_op_column)

        # Then return the op-records from the new out-column.
        if len(out_op_column.op_records) == 1:
            return out_op_column.op_records[0]
        else:
            return out_op_column.op_records

    def check_input_completeness(self):
        """
        Checks whether this Component is "input-complete" and stores the result in self.input_complete.
        Input-completeness is reached (only once and then it stays that way) if all API-methods of this component
        (whose `must_be_complete` field is not set to False) have at least one op-rec-column completed.

        Returns:
            Optional[dict]: A space-dict if the Component is input-complete, None otherwise.
        """
        assert self.input_complete is False
        space_dict = dict()

        self.input_complete = True
        # Loop through all API methods.
        for method_name, api_method_rec in self.api_methods.items():
            # This API method doesn't have to be completed, ignore and don't add it to space_dict.
            if api_method_rec.must_be_complete is False:
                continue

            # Get the spaces of each op-record in the columns.
            # If one of the columns is complete, Component is complete.
            for in_op_col in api_method_rec.in_op_columns:
                spaces = [op_rec.space for op_rec in in_op_col.op_records]
                # All Spaces are defined -> Store list of Spaces (for this column) in return dict.
                if all(s is not None for s in spaces):
                    space_dict[method_name] = spaces
                    break
            # None of the columns is complete. Return as "incomplete".
            else:
                if len(api_method_rec.in_op_columns) > 0:
                    self.input_complete = False
                    return None

        return space_dict

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

    def when_input_complete(self, input_spaces, action_space, device=None, summary_regexp=None):
        """
        Wrapper that calls both `create_variables` and `assert_input_spaces` in sequence and passes the dict with
        the input_spaces for each in-Socket (kay=Socket's name) as parameter.

        Args:
            input_spaces (Dict[str,Space]): A dict with Space/shape information.
                keys=in-Socket name (str); values=the associated Space
            action_space (Space): The action Space of the Agent/GraphBuilder. Can be used to construct and connect
                more Components (which rely on this information). This eliminates the need to pass the action Space
                information into many Components' constructors.
            device (str): The device to use for the variables generated.
            summary_regexp (Optional[str]): A regexp (str) that defines, which summaries should be generated
                and registered.
        """
        # Store the summary_regexp to use.
        self.summary_regexp = summary_regexp

        # Allow the Component to check its input Space.
        self.check_input_spaces(input_spaces, action_space)
        # Allow the Component to create all its variables.
        if get_backend() == "tf":
            with tf.device(device):
                with tf.variable_scope(self.global_scope):
                    self.create_variables(input_spaces, action_space)

        # Add all created variables up the parent/container hierarchy.
        self.propagate_variables()

    def check_input_spaces(self, input_spaces, action_space):
        """
        Should check on the nature of all in-Sockets Spaces of this Component. This method is called automatically
        by the Model when all these Spaces are know during the Model's build time.

        Args:
            input_spaces (Dict[str,Space]): A dict with Space/shape information.
                keys=in-Socket name (str); values=the associated Space
            action_space (Space): The action Space of the Agent/GraphBuilder. Can be used to construct and connect
                more Components (which rely on this information). This eliminates the need to pass the action Space
                information into many Components' constructors.
        """
        pass

    def create_variables(self, input_spaces, action_space):
        """
        Should create all variables that are needed within this component,
        unless a variable is only needed inside a single _graph_fn-method, in which case,
        it should be created there.
        Variables must be created via the backend-agnostic self.get_variable-method.

        Note that for different scopes in which this component is being used, variables will not(!) be shared.

        Args:
            input_spaces (Dict[str,Space]): A dict with Space/shape information.
                keys=in-Socket name (str); values=the associated Space
            action_space (Space): The action Space of the Agent/GraphBuilder. Can be used to construct and connect
                more Components (which rely on this information). This eliminates the need to pass the action Space
                information into many Components' constructors.
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
                     from_space=None, add_batch_rank=False, flatten=False):
        """
        Generates or returns a variable to use in the selected backend.
        The generated variable is automatically registered in this component's (and all parent components')
        variable-registry under its global-scoped name.

        Args:
            name (str): The name under which the variable is registered in this component.
            shape (Optional[tuple]): The shape of the variable. Default: empty tuple.
            dtype (Union[str,type]): The dtype (as string) of this variable.
            initializer (Optional[any]): Initializer for this variable.
            trainable (bool): Whether this variable should be trainable.
            from_space (Optional[Space,str]): Whether to create this variable from a Space object
                (shape and dtype are not needed then). The Space object can be given directly or via the name
                of the in-Socket holding the Space.
            add_batch_rank (Optional[bool,int]): If from_space is given and is True, will add a 0th rank (None) to
                the created variable. If it is an int, will add that int instead of None.
                Default: False.
            flatten (bool): Whether to produce a FlattenedDataOp with auto-keys.

        Returns:
            DataOp: The actual variable (dependent on the backend) or - if from
                a ContainerSpace - a FlattenedDataOp or ContainerDataOp depending on the Space.
        """
        # Called as getter.
        if shape is None and initializer is None and from_space is None:
            if name not in self.variables:
                raise KeyError("Variable with name '{}' not found in registry of Component '{}'!".
                               format(name, self.name))
            # TODO: Maybe try both the pure name AND the name with global-scope in front.
            return self.variables[name]

        # Called as setter.
        var = None

        # We are creating the variable using a Space as template.
        if from_space is not None:
            # Variables should be returned in a flattened OrderedDict.
            if flatten:
                var = from_space.flatten(mapping=lambda k, primitive: primitive.get_tensor_variable(
                    name=name + k, add_batch_rank=add_batch_rank, trainable=trainable, initializer=initializer))
            # Normal, nested Variables from a Space (container or primitive).
            else:
                var = from_space.get_tensor_variable(name=name, add_batch_rank=add_batch_rank, trainable=trainable,
                                                     initializer=initializer)
        # Direct variable creation (using the backend).
        elif get_backend() == "tf":
            # Provide a shape, if initializer is not given or it is an actual Initializer object (rather than an array
            # of fixed values, for which we then don't need a shape as it comes with one).
            if initializer is None or isinstance(initializer, tf.keras.initializers.Initializer):
                shape = tuple((() if add_batch_rank is False else
                               (None,) if add_batch_rank is True else (add_batch_rank,)) + (shape or ()))
            else:
                shape = None

            var = tf.get_variable(
                name=name, shape=shape, dtype=util.dtype(dtype), initializer=initializer, trainable=trainable
            )
        elif get_backend() == "tf-eager":
            shape = tuple((() if add_batch_rank is False else (None,) if add_batch_rank is True else (add_batch_rank,))
                          + (shape or ()))

            var = eager.Variable(
                name=name, shape=shape, dtype=util.dtype(dtype), initializer=initializer, trainable=trainable
            )

        # Registers the new variable with this Component.
        key = ((self.global_scope + "/") if self.global_scope else "") + name
        # Container-var: Save individual Variables.
        # TODO: What about a var from Tuple space?
        if isinstance(var, OrderedDict):
            for sub_key, v in var.items():
                self.variables[key + sub_key] = v
        else:
            self.variables[key] = var

        return var

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
            collections = kwargs.pop("collections", None) or tf.GraphKeys.TRAINABLE_VARIABLES
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
                            ret[re.sub(r'/', custom_scope_separator, lookup)] = v
                        else:
                            ret[re.sub(r'^.+/', "", lookup)] = v
                return ret
            # Return only variables of this Component by name.
            else:
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
        # Skip non matching summaries.
        if self.summary_regexp is not None and not re.search(self.summary_regexp, global_name):
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

    def define_api_method(self, name, func=None, must_be_complete=True, **kwargs):
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

        Keyword Args:
            flatten_ops (bool,Set[int]): See `self.call` for details.
            split_ops (bool,Set[int]): See `self.call` for details.
            add_auto_key_as_first_param (bool): See `self.call` for details.
        """
        # There already is an API-method with that name.
        if name in self.api_methods:
            raise RLGraphError("API-method with name '{}' already defined!".format(name))
        # There already is another object property with that name (avoid accidental overriding).
        elif getattr(self, name, None) is not None:
            raise RLGraphError("Component '{}' already has a property called '{}'. Cannot define an API-method with "
                            "the same name!".format(self.name, name))
        # Do not build this API as per ctor instructions.
        elif name in self.switched_off_apis:
            return

        func_type = util.get_method_type(func)

        # Function is a graph_fn: Build a simple wrapper API-method around it and name it `name`.
        if func_type == "graph_fn":

            def api_method(self_, *inputs):
                func_ = getattr(self_, func.__name__)
                return self_.call(func_, *inputs, **kwargs)

        # Function is a (custom) API-method. Register it with this Component.
        else:
            api_method = func

        setattr(self, name, api_method.__get__(self, self.__class__))
        setattr(api_method, "__self__", self)
        setattr(api_method, "__name__", name)

        self.api_methods[name] = APIMethodRecord(getattr(self, name), component=self, must_be_complete=must_be_complete)

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
            component.parent_component = self
            self.sub_components[component.name] = component

            # Fix the sub-component's (and sub-sub-component's etc..) scope(s).
            self.propagate_scope(component)

            # Should we expose some API-methods of the child?
            for api_method_name, api_method_rec in component.api_methods.items():
                if api_method_name in expose_apis:
                    def exposed_api_method_wrapper(self_, *inputs):
                        return self_.call(api_method_rec.method, *inputs)
                    self.define_api_method(api_method_name, exposed_api_method_wrapper)
                    # Add the sub-component's API-registered methods to ours.
                    #self.defined_externally.add(component.scope + "-" + api_method_name)

    def propagate_scope(self, sub_component):
        """
        Fixes all the sub-Component's (and its sub-Component's) global_scopes.

        Args:
            sub_component (Optional[Component]): The sub-Component object whose global_scope needs to be updated.
                Use None for this Component itself.
        """
        if sub_component is None:
            sub_component = self
        elif self.global_scope:
            sub_component.global_scope = self.global_scope + (
                ("/" + sub_component.scope) if sub_component.scope else "")
        # Recurse.
        for sc in sub_component.sub_components.values():
            sub_component.propagate_scope(sc)

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

    def copy(self, name=None, scope=None, device=None, trainable=None, global_component=False):
        """
        Copies this component and returns a new component with possibly another name and another scope.
        The new component has its own variables (they are not shared with the variables of this component as they
        will be created after this copy anyway, during the build phase).
        and is initially not connected to any other component. However, the Sockets of this component and their names
        are being copied (but without their connections).

        Args:
            name (str): The name of the new Component. If None, use the value of scope.
            scope (str): The scope of the new Component. If None, use the same scope as this component.
            device (str): The device of the new Component. If None, use the same device as this one.
            trainable (Optional[bool]): Whether to make all variables in this component trainable or not.
                Use None for no specific preference.
            global_component (Optional[bool]): Whether the new Component is global or not. If None, use the same
                setting as this one.

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
        if global_component is None:
            global_component = self.global_component

        # Simply deepcopy self and change name and scope.
        new_component = copy.deepcopy(self)
        new_component.name = name
        new_component.scope = scope
        # Change global_scope for the copy and all its sub-components.
        new_component.global_scope = scope
        new_component.propagate_scope(sub_component=None)
        new_component.device = device
        new_component.trainable = trainable
        new_component.global_component = global_component
        new_component.parent_component = None  # erase the parent pointer

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

    def __str__(self):
        return "{}('{}' api={})".format(type(self).__name__, self.name, str(list(self.api_methods.keys())))
