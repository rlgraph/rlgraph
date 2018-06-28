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

from collections import Hashable, OrderedDict
import copy
import inspect
import numpy as np
import re
import uuid

from yarl import YARLError, get_backend, Specifiable
from yarl.utils.ops import SingleDataOp, DataOpDict, DataOpRecord, APIMethodRecord, DataOpRecordColumn
from yarl.utils import util
from yarl.spaces import Space

# Some settings flags.
# Whether to expose only in-Sockets (in calls to add_component(s)).
CONNECT_INS = 0x1
# Whether to expose only out-Sockets (in calls to add_component(s))
CONNECT_OUTS = 0x2
# Whether to expose all in/out-Sockets (in calls to add_component(s))
CONNECT_ALL = 0x4

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
    def __init__(self, *components_to_add, **kwargs):
        """
        Args:
            components_to_add (Component): Specification dicts for sub-Components to be added to this one.

        Keyword Args:
            name (str): The name of this Component. Names of sub-components within a containing component
                must be unique. Names are used to label exposed Sockets of the containing component.
                If name is empty, use scope as name (as last resort).
            scope (str): The scope of this Component for naming variables in the Graph.
            device (str): Device this component will be assigned to. If None, defaults to CPU.
            global_component (bool): In distributed mode, this flag indicates if the component is part of the
                shared global model or local to the worker. Defaults to False and will be ignored if set to
                True in non-distributed mode.
            is_core (bool): Whether this Component is the Model's core Component.

            sub_components (Component): An alternative way to pass in a list of Components to be added as
                sub-components to this one.
            inputs (List[str]): A list of in-Sockets to be defined for this Component.
            outputs (List[str]): A list of out-Sockets to be defined for this Component.
            connections (list): A list of connection specifications to be made between our Sockets and/or the different
                sub-Components.
        """

        # Scope if used to create scope hierarchies inside the Graph.
        self.scope = kwargs.pop("scope", "")
        assert re.match(r'^[\w\-]*$', self.scope), \
            "ERROR: scope {} does not match scope-pattern! Needs to be \\w or '-'.".format(self.scope)
        # The global scope string defining the exact nexted position of this Component in the Graph.
        # e.g. "/core/component1/sub-component-a"
        self.global_scope = self.scope
        # Names of sub-components that exist (parallelly) inside a containing component must be unique.
        self.name = kwargs.pop("name", self.scope)  # if no name given, use scope
        self.device = kwargs.pop("device", None)
        self.global_component = kwargs.pop("global_component", False)
        self.is_core = kwargs.pop("is_core", False)

        # A group ID counter for parallelly incoming op records that need to be processed together.
        #self.op_group_id = 0

        #api_methods = util.force_list(kwargs.pop("api_methods", []))
        #outputs = util.force_list(kwargs.pop("outputs", []))
        #connections = kwargs.pop("connections", [])

        assert not kwargs, "ERROR: kwargs ({}) still contains items!".format(kwargs)

        # Dict of sub-components that live inside this one (key=sub-component's scope).
        self.sub_components = OrderedDict()

        # Simple list of all GraphFunction objects that have ever been added to this Component.
        #self.graph_fns = list()

        # Keep track of whether this Component has already been added to another Component and throw error
        # if this is done twice. Each Component can only be added once to a parent Component.
        self.parent_component = None  # type: Component

        # Dicts holding information about which op-record-tuples go via which API methods into this Component
        # and come out of it.
        # keys=API method name; values=APIMethodRecord
        self.api_methods = self.get_api_methods()
        self.outputs = dict()
        ### dict with key=API method name; value=list of in-Spaces that would go into the method.

        # This Component's Socket objects by functionality.
        #self.input_sockets = list()  # the exposed in-Sockets
        #self.output_sockets = list()  # the exposed out-Sockets
        #self.internal_sockets = list()  # internal (non-exposed) in/out-Sockets (e.g. used to connect 2 GraphFunctions)

        # Whether we know already all our in-Sockets' Spaces.
        # Only then can we create our variables. Model will do this.
        self.input_complete = False

        # A set of in-Socket names for which it is ok, not to be connected after the
        # YARL-meta graph is passed on for building.
        #self.unconnected_sockets_in_meta_graph = set()

        # Contains sub-Components of ours that do not have in-Sockets.
        #self.no_input_sub_components = set()
        # Contains our GraphFunctions that have no in-Sockets.
        #self.no_input_graph_fns = set()

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

        #self.define_inputs(*api_methods)
        #self.define_outputs("_variables", *outputs)

        # Adds the default "_variables" out-Socket responsible for sending out all our variables.
        #self.add_graph_fn(None, "_variables", self._graph_fn__variables, flatten_ops=False)

        # Add the sub-components.
        #components_to_add = components_to_add if len(components_to_add) > 0 else kwargs.get("sub_components", [])
        #for sub_component_spec in components_to_add:
        #    self.add_component(Component.from_spec(sub_component_spec))

        ## Add the connections.
        #for connection in connections:
        #    self.connect(connection[0], connection[1])

    def get_api_methods(self):
        ret = dict()
        # look for all our API methods (those that use the `call` method).
        for member in inspect.getmembers(self):
            name, method = (member[0], member[1])
            if name[0] != "_" and name != "define_api_method" and callable(method) and \
                    re.search(r'\bself\.call\(', inspect.getsource(method)):
                ret[name] = APIMethodRecord(method, component=self)
        return ret

    def call(self, method, params=None, flatten_ops=False, split_ops=False, add_auto_key_as_first_param=False):
        """
        Performs either:
        a) An assembly run through a graph_fn (without actually calling it) using the op-record column
            provided.
        b) An actual dynamic run through a graph_fn with the data (in the op_columns? TODO: fix for pytorch).

        Args:
            method (callable): The method (graph_fn or API method) to call.
            params (DataOpRecordColumn): The DataOpRecordColumn to be pushed through the graph_fn.
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

        Returns:
            The
        """
        in_op_recs = util.force_list(params)

        # Owner of method:
        method_owner = method.__self__

        # method is an API method.
        if method.__name__ in method_owner.api_methods:
            api_method_rec = method_owner.api_methods[method.__name__]
            # Create op-record column to call API method with.
            in_op_column = DataOpRecordColumn(op_records=len(in_op_recs))
            api_method_rec.in_op_columns.append(in_op_column)

            # Link from in_op_recs into the new column.
            for i, op_rec in enumerate(in_op_recs):
                op_rec.next.add(in_op_column.op_records[i])
            # Now actually call the API method with that column.
            return method(*in_op_column.op_records)

        # Method is a graph_fn.
        else:
            # Create 2 op-record columns, one going into the graph_fn and one getting out of there and link
            # them together via the graph_fn (w/o calling it).
            out_graph_fn_column = DataOpRecordColumn(1)  # TODO: hardcoded: 1 return value
            in_graph_fn_column = DataOpRecordColumn(
                len(in_op_recs), graph_fn=method, out_graph_fn_column=out_graph_fn_column,
                flatten_ops=flatten_ops, split_ops=split_ops, add_auto_key_as_first_param=add_auto_key_as_first_param,
                component=self
            )
            # Link from in_op_recs into the new column.
            for i, op_rec in enumerate(in_op_recs):
                op_rec.next.add(in_graph_fn_column.op_records[i])

            if len(out_graph_fn_column.op_records) == 1:
                return out_graph_fn_column.op_records[0]
            else:
                return out_graph_fn_column.op_records

    def when_input_complete(self, input_spaces, action_space, summary_regexp=None):
        """
        Wrapper that calls both `create_variables` and `assert_input_spaces` in sequence and passes the dict with
        the input_spaces for each in-Socket (kay=Socket's name) as parameter.

        Args:
            input_spaces (Dict[str,Space]): A dict with Space/shape information.
                keys=in-Socket name (str); values=the associated Space
            action_space (Space): The action Space of the Agent/GraphBuilder. Can be used to construct and connect
                more Components (which rely on this information). This eliminates the need to pass the action Space
                information into many Components' constructors.
            summary_regexp (Optional[str]): A regexp (str) that defines, which summaries should be generated
                and registered.
        """
        # Store the summary_regexp to use.
        self.summary_regexp = summary_regexp

        # Allow the Component to check its input Space.
        self.check_input_spaces(input_spaces, action_space)
        # Allow the Component to create all its variables.
        if get_backend() == "tf":
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
                    name=name + k, add_batch_rank=add_batch_rank, trainable=trainable))
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
        if isinstance(var, OrderedDict):
            for sub_key, v in var.items():
                self.variables[key+sub_key] = v
        else:
            self.variables[key] = var

        return var

    def get_variables(self, *names,  **kwargs):
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
                return self.get_variables_by_name(custom_scope_separator, global_scope, names)

    def get_variables_by_name(self, custom_scope_separator, global_scope, names):
        """
        Retrieves this components variables by name.

        Args:
            names (List[str]): List of names of Variable to return.
            custom_scope_separator (str): The separator to use in the returned dict for scopes.
                Default: '/'.
            global_scope (bool): Whether to use keys in the returned dict that include the global-scopes of the
                Variables. Default: False.

        Returns:
            dict: Dict containing the requested names as keys and variables as values.
        """
        variables = dict()
        for name in names:
            global_scope_name = ((self.global_scope + "/") if self.global_scope else "") + name
            if name in self.variables:
                variables[re.sub(r'/', custom_scope_separator, name)] = self.variables[name]
            elif global_scope_name in self.variables:
                if global_scope:
                    variables[re.sub(r'/', custom_scope_separator, global_scope_name)] = self.variables[global_scope_name]
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
        name = "summaries/"+name
        # Get global name.
        global_name = ((self.global_scope + "/") if self.global_scope else "") + name
        # Skip non matching summaries.
        if not re.search(self.summary_regexp, global_name):
            return

        summary = None
        if get_backend() == "tf":
            ctor = getattr(tf.summary, type_)
            summary = ctor(name, values)

        # Registers the new summary with this Component.
        if global_name in self.summaries:
            raise YARLError("ERROR: Summary with name '{}' already exists in {}'s summary "
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
            #if self.variables[key] is not self.parent_component.variables[key]:
            raise YARLError("ERROR: Summary registry of '{}' already has a summary under key '{}'!". \
                            format(self.parent_component.name, key_))
        self.parent_component.summaries[key_] = self.summaries[key_]

        # Recurse up the container hierarchy.
        self.parent_component.propagate_summary(key_)

    def define_api_method(self, name, graph_fn, **kwargs):
        """
        Creates a very basic graph_fn based API method for this Component.
        Alternative way for defining an API method directly in the Component via def.

        Args:
            name (str): The name of the API method to create.
            graph_fn (callable): The graph_fn to wrap.

        Keyword Args:
            flatten_ops (bool,Set[int]): See `self.call` for details.
            split_ops (bool,Set[int]): See `self.call` for details.
            add_auto_key_as_first_param (bool): See `self.call` for details.
        """
        assert name not in self.api_methods and getattr(self, name, None) is None

        def api_method(*inputs):
            return self.call(graph_fn, params=inputs, **kwargs)

        self.__setattr__(name, api_method)
        self.api_methods[name] = APIMethodRecord(api_method, component=self)

    def OBSOLETE_define_inputs(self, *sockets, **kwargs):
        """
        Calls add_sockets with type="in" and then connects each of the Sockets with the given
        Space.

        Args:
            *sockets (List[Socket]): The list of in-Sockets to add.

        Keyword Args:
            space (Optional[Space]): The Space to connect to the in-Sockets.
        """
        self.add_sockets(*sockets, type="in")
        space = kwargs.pop("space", None)
        assert not kwargs

        if space is not None:
            for socket in sockets:
                self.connect(space, self.get_socket(socket).name)

    def OBSOLETE_define_outputs(self, *sockets):
        """
        Calls add_sockets with type="out".

        Args:
            *sockets (List[Socket]): The list of out-Sockets to add.
        """
        self.add_sockets(*sockets, type="out")

    def OBSOLETE_add_socket(self, name, value=None, **kwargs):
        """
        Adds new input/output Sockets to this component.

        Args:
            name (str): The name for the Socket to be created.
            value (Optional[numeric]): The constant value of the Socket (if this should be a constant-value Socket).

        Keyword Args:
            type (str): Either "in" or "out". If no type given, try to infer type by the name.
                If name contains "input" -> type is "in", if name contains "output" -> type is "out".
            internal (bool): True if this is an internal (non-exposed) Socket (e.g. used for connecting
                2 GraphFunctions directly with each other).

        Raises:
            YARLError: If type is not given and cannot be inferred.
        """
        type_ = kwargs.pop("type", kwargs.pop("type_", None))
        internal = kwargs.pop("internal", False)
        assert not kwargs

        # Make sure we don't add two sockets with the same name.
        all_names = [sock.name for sock in self.input_sockets + self.output_sockets + self.internal_sockets]

        # Try to infer type by socket name (if name contains 'input' or 'output').
        if type_ is None:
            mo = re.search(r'([\d_]|\b)(in|out)put([\d_]|\b)', name)
            if mo:
                type_ = mo.group(2)
            if type_ is None:
                raise YARLError("ERROR: Cannot infer socket type (in/out) from socket name: '{}'!".format(name))

        if name in all_names:
            raise YARLError("ERROR: Socket of name '{}' already exists in component '{}'!".format(name, self.name))

        sock = Socket(name=name, component=self, type_=type_)

        if internal:
            self.internal_sockets.append(sock)
        elif type_ == "in":
            self.input_sockets.append(sock)
        else:
            self.output_sockets.append(sock)

        # A constant value Socket.
        if value is not None:
            op = SingleDataOp(constant_value=value)
            sock.connect_from(op)

    def OBSOLETE_add_sockets(self, *socket_names, **kwargs):
        """
        Adds new input/output Sockets to this component.

        Args:
            *socket_names (List[str]): List of names for the Sockets to be created.

        Keyword Args:
            type (str): Either "in" or "out". If no type given, try to infer type by the name.
                If name contains "input" -> type is "in", if name contains "output" -> type is "out".
            internal (bool): True if this is an internal (non-exposed) Socket (e.g. used for connecting
                2 GraphFunctions directly with each other).

        Raises:
            YARLError: If type is not given and cannot be inferred.
        """
        for name in socket_names:
            self.add_socket(name, **kwargs)

    def OBSOLETE_rename_socket(self, old_name, new_name):
        """
        Renames an already existing in- or out-Socket of this Component.

        Args:
            old_name (str): The current name of the Socket.
            new_name (str): The new name of the Socket. No other Socket with this name must exist in this Component.
        """
        all_sockets = self.input_sockets + self.output_sockets + self.internal_sockets
        socket = self.get_socket_by_name(self, old_name)  # type: Socket
        assert socket in all_sockets,\
            "ERROR: Socket with name '{}' not found in Component '{}'!".format(old_name, self.name)
        # Make sure the new name is not taken yet.
        socket_with_new_name = self.get_socket_by_name(self, new_name)
        assert socket_with_new_name is None,\
            "ERROR: Socket with name '{}' already exists in Component '{}'!".format(new_name, self.name)
        socket.name = new_name

    def OBSOLETE_remove_socket(self, socket_name):
        """
        Removes an in- or out-Socket from this Component. The Socket to be removed must not have any connections yet
        from other (external) Components.

        Args:
            socket_name (str): The name of the Socket to be removed.
        """
        socket = self.get_socket_by_name(self, socket_name)  # type: Socket
        assert socket, "ERROR: Socket with name '{}' not found in Component '{}'!".format(socket_name, self.name)
        assert socket not in self.internal_sockets,\
            "ERROR: Cannot remove an internal Socket ({}) in Component {}!".format(socket.name, self.name)

        # Make sure this Socket is without connections.
        if socket.type == "in" and len(socket.incoming_connections) > 0:
            raise YARLError("ERROR: Cannot remove an in-Socket ('{}') that already has incoming connections!".
                            format(socket.name))
        elif socket.type == "out" and len(socket.outgoing_connections) > 0:
            raise YARLError("ERROR: Cannot remove an out-Socket ('{}') that already has outgoing connections!".
                            format(socket.name))

        # in-Socket
        if socket in self.input_sockets:
            self.input_sockets.remove(socket)
        # out-Socket
        else:
            self.output_sockets.remove(socket)

    def OBSOLETE_add_graph_fn(self, inputs, outputs, method,
                     flatten_ops=False, split_ops=False, add_auto_key_as_first_param=False):
        """
        Links a set (A) of sockets via a graph_fn to a set (B) of other sockets via a graph_fn function.
        Any socket in B is thus linked back to all sockets in A (requires all A sockets).

        Args:
            inputs (Optional[str,List[str]]): List or single input Socket specifiers to link from.
            outputs (Optional[str,List[str]]): List or single output Socket specifiers to link to.
            method (Union[str,callable]): The name of the template method to use for linking
                (without the _graph_ prefix) or the method itself (callable).
                The `method`'s signature and number of return values has to match the
                number of input- and output Sockets provided here.
            flatten_ops (bool,Set[str]): Passed to GraphFunction's c'tor. See GraphFunction for details.
                Default: False.
            split_ops (bool,Set[str]): Passed to GraphFunction's c'tor. See GraphFunction for details.
                Default: False.
            add_auto_key_as_first_param (bool): Passed to GraphFunction's c'tor. See GraphFunction for details.
                Default: False.

        Raises:
            YARLError: If method_name is None and not exactly one member method found that starts with `_graph_`.
        """
        inputs = util.force_list(inputs)
        outputs = util.force_list(outputs)

        # TODO: Make it possible to call other graph_fns within a graph_fn and to call a sub-component's graph_fn within a graph_fn.
        # TODO: Sanity check the tf methods signature (and return values from docstring?).
        # Compile a list of all needed Sockets and create internal ones if they do not exist yet.
        input_sockets = [self.get_socket(s, create_internal_if_not_found=True) for s in inputs]
        output_sockets = [self.get_socket(s, create_internal_if_not_found=True) for s in outputs]

        # Add the graph_fn record to all input and output sockets.
        graph_fn = GraphFunction(
            method=method,
            component=self,
            input_sockets=input_sockets,
            output_sockets=output_sockets,
            flatten_ops=flatten_ops,
            split_ops=split_ops,
            add_auto_key_as_first_param=add_auto_key_as_first_param,
        )
        self.graph_fns.append(graph_fn)

        # Connect the graph_fn to all the given Sockets.
        for input_socket in input_sockets:
            input_socket.connect_to(graph_fn)
        for output_socket in output_sockets:
            output_socket.connect_from(graph_fn)

        # If the GraphFunction has no api_methods or only constant value api_methods, we need to treat it separately during
        # the build. Exception: _variables, which has to be built last after all our sub-components are done
        # creating their variables and pushing them up to this Component.
        # FixMe: What if an internal graph_fn input is blocked with a constant value? Need test case for that.
        if len(input_sockets) == 0 and graph_fn.name != "_variables":
            self.no_input_graph_fns.add(graph_fn)

    def OBSOLETE_change_graph_fn_options(self, name, flatten_ops=None, split_ops=None, add_auto_key_as_first_param=None):
        """
        Changes the options of one of our graph_fns.

        Args:
            name (str): The name of the graph_fn to change.
            flatten_ops (Optional[bool]): If not None, change this option of the graph_fn to the given value.
            split_ops (Optional[bool]): If not None, change this option of the graph_fn to the given value.
            add_auto_key_as_first_param (Optional[bool]): If not None, change this option of the graph_fn
                to the given value.
        """
        for graph_fn in self.graph_fns:  # type: GraphFunction
            if graph_fn.name == name:
                if flatten_ops is not None:
                    graph_fn.flatten_ops = flatten_ops
                if split_ops is not None:
                    graph_fn.split_ops = split_ops
                if add_auto_key_as_first_param is not None:
                    graph_fn.add_auto_key_as_first_param = add_auto_key_as_first_param

    def add_components(self, *components):
        """
        Adds sub-components to this one.

        Args:
            components (List[Component]): The list of Component objects to be added into this one.
        """
        for component in components:
            # Make sure no two components with the same name are added to this one (own scope doesn't matter).
            if component.name in self.sub_components:
                raise YARLError("ERROR: Sub-Component with name '{}' already exists in this one!".
                                format(component.name))
            # Make sure each Component can only be added once to a parent/container Component.
            elif component.parent_component is not None:
                raise YARLError("ERROR: Sub-Component with name '{}' has already been added once to a container "
                                "Component! Each Component can only be added once to a parent.".format(component.name))
            component.parent_component = self
            self.sub_components[component.name] = component

            # Fix the sub-component's (and sub-sub-component's etc..) scope(s).
            self.propagate_scope(component)

    def propagate_scope(self, sub_component):
        """
        Fixes all the sub-Component's (and its sub-Component's) global_scopes.

        Args:
            sub_component (Optional[Component]): The sub-Component object whose global_scope needs to be updated.
                USe None for this Component itself.
        """
        if sub_component is None:
            sub_component = self
        elif self.global_scope:
            sub_component.global_scope = self.global_scope + (("/"+sub_component.scope) if sub_component.scope else "")
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
                    raise YARLError("ERROR: Variable registry of '{}' already has a variable under key '{}'!".\
                        format(self.parent_component.name, key))
            self.parent_component.variables[key] = self.variables[key]

        # Recurse up the container hierarchy.
        self.parent_component.propagate_variables(keys)

    def copy(self, name=None, scope=None, device=None, global_component=False):
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
            global_component (Optional[bool]): Whether the new Component is global or not. If None, use the same
                setting as this one.

        Returns:
            Component: The copied component object.
        """
        if scope is None:
            scope = self.scope
        if name is None:
            name = scope
        if device is None:
            device = self.device
        if global_component is None:
            global_component = self.global_component

        # Simply deepcopy self and change name and scope.
        new_component = copy.deepcopy(self)
        new_component.name = name
        new_component.scope = scope
        # Change global_scope for copy and all its sub-components.
        new_component.global_scope = scope
        new_component.propagate_scope(sub_component=None)
        new_component.device = device
        new_component.global_component = global_component
        new_component.parent_component = None  # erase the parent pointer

        # Then cut all the new_component's outside connections (no need to worry about the other side as
        # they were not notified of the copied Sockets).
        for socket in new_component.input_sockets:
            socket.incoming_connections = list()
        for socket in new_component.output_sockets:
            socket.outgoing_connections = list()

        return new_component

    def OBSOLETE_connect(self, from_, to_):
        """
        Generic connection method to create connections: between
        - a Socket (from_) and another Socket (to_).
        - a Socket and a Space (or the other way around).
        `from_` and `to_` can be directly Socket objects but also Socket-specifiers. See `self.get_socket` for details
        on possible ways of specifying a socket (by string, component-name, etc..).

        Also, `from_` and/or `to_` may be Component objects. In that case, all out-Sockets of `from_` will be
        connected to the respective in-Sockets of `to_`.

        Note that more specialised methods are available and should be used to increase code
        readability.

        Args:
            from_ (any): The specifier of the connector (e.g. incoming Socket, an incoming Space).
            to_ (any): The specifier of the connectee (e.g. another Socket).
        """
        self._connect_disconnect(from_, to_, disconnect_=False)

    def OBSOLETE_disconnect(self, from_, to_):
        """
        Removes a connection between:
        - a Socket (from_) and another Socket (to_).
        - a Socket and a Space (or the other way around).
        `from_` and `to_` can be directly Socket objects but also Socket-specifiers. See `self.get_socket` for details
        on possible ways of specifying a socket (by string, component-name, etc..).
        Also, `from_` and/or `to_` may be Component objects. In that case, all out-Socket connections of `from_` to all
        in-Socket connections of `to_` are cut.

        Args:
            from_ (any): See `self.connect` for details.
            to_ (any): See `self.connect` for details.
        """
        self._connect_disconnect(from_, to_, disconnect_=True)

    def OBSOLETE__connect_disconnect(self, from_, to_, disconnect_=False):
        """
        Actual private implementer for `connect` and `disconnect`.

        Args:
            from_ (any): See `self.connect` for details.
            to_ (any): See `self.connect` for details.
            disconnect_ (bool): Only used internally. Whether to actually disconnect (instead of connect).
        """
        # Connect a Space (other must be Socket).
        # Also, there are certain restrictions for the Socket's type.
        if isinstance(from_, (Space, dict, SingleDataOp)) or \
                (not isinstance(from_, str) and isinstance(from_, Hashable) and from_ in Space.__lookup_classes__):
            return self.connect_space_to_socket(from_, to_, disconnect_)
        # TODO: Special case: Never really done yet: Connecting an out-Socket to a Space (usually: action-space).
        elif isinstance(to_, (Space, dict)) or \
                (not isinstance(to_, str) and isinstance(to_, Hashable) and to_ in Space.__lookup_classes__):
            return self.connect_out_socket_to_space(from_, to_, disconnect_)
        # to_ is Component.
        if isinstance(to_, Component):
            # Both from_ and to_ are Components: Connect them Socket-by-Socket and return.
            if isinstance(from_, Component):
                # -1 for the special "_variables" Socket, which should not be connected.
                return self.connect_component_to_component(from_, to_)  # , label
            # Get the 1 in-Socket of to_.
            to_socket_obj = to_.get_socket(type_="in")
        # to_ is Socket specifier.
        else:
            to_socket_obj = self.get_socket(to_)

        # from_ is Component. Get the 1 out-Socket of from_.
        if isinstance(from_, Component):
            from_socket_obj = from_.get_socket(type_="out")
        # from_ is a constant value -> create SingleDataOp with constant_value defined and connect from it.
        elif isinstance(from_, (int, float, bool, np.ndarray)):
            assert disconnect_ is False  # hard assert
            to_socket_obj.connect_from(SingleDataOp(constant_value=from_))
            return
        # from_ is a DataOpRecord with a Socket field. Connect the Sockets,
        # and make sure that the from-Socket pushes that particular record into
        # the out-Socket's record.
        elif isinstance(from_, DataOpRecord):
            assert from_.socket is not None
            from_socket_obj = from_.socket
            # Create a new op-rec for to_socket_obj.
            to_op_rec = DataOpRecord(filter_by_socket_name=from_socket_obj.name, socket=to_socket_obj)
            from_.push_op_into_recs.add(to_op_rec)
        # Regular Socket->Socket connection.
        else:
            from_socket_obj = self.get_socket(from_)

        # (Dis)connect the two Sockets in both ways.
        if disconnect_:
            from_socket_obj.disconnect_to(to_socket_obj)
            to_socket_obj.disconnect_from(from_socket_obj)
        else:
            from_socket_obj.connect_to(to_socket_obj)
            to_socket_obj.connect_from(from_socket_obj)

    def OBSOLETE_connect_space_to_socket(self, input_space, in_socket, disconnect=False):
        """
        Connects a space to an input-socket.
        Args:
            input_space (Union[Space, dict, SingleDataOp]): Input space or space definition.
            in_socket (Socket): Socket to connect.
            disconnect (bool): Only used internally. Whether to actually disconnect (instead of connect).
        """
        input_space = input_space if isinstance(input_space, (Space, SingleDataOp)) else Space.from_spec(input_space)
        socket_obj = self.get_socket(in_socket)
        assert socket_obj.type == "in", "ERROR: Cannot connect a Space to an 'out'-Socket!"
        if not disconnect:
            socket_obj.connect_from(input_space)
        else:
            socket_obj.disconnect_from(input_space)

    def OBSOLETE_connect_out_socket_to_space(self, out_socket, space, disconnect=False):
        """
        Connects an out socket to a space, e.g. an action space.

        Args:
            out_socket (Socket): Out-socket to connect.
            space (Union[Space, dict, SingleDataOp]): Space or space definition.
            disconnect (bool): Only used internally. Whether to actually disconnect (instead of connect).
        """
        to_space = space if isinstance(space, Space) else Space.from_spec(space)
        socket_obj = self.get_socket(out_socket)
        assert socket_obj.type == "out", "ERROR: Cannot connect an 'in'-Socket to a Space!"
        if not disconnect:
            socket_obj.connect_to(to_space)
        else:
            socket_obj.disconnect_to(to_space)

    def OBSOLETE_connect_component_to_component(self, from_component, to_component):
        """
        Connects a component to another component via iterating over all their sockets.
        Components must have the same number of sockets.

        Args:
            from_component (Component): Component from which to connect/
            to_component (Component): Component to connect to.
        """
        assert len(to_component.output_sockets) - 1 == len(from_component.input_sockets), \
            "ERROR: Can only connect two Components if their Socket numbers match! {} has {} out-Sockets while " \
            "{} has {} in-Sockets.".format(from_component.name, len(from_component.output_sockets), to_component.name,
                                           len(to_component.input_sockets))
        out_sock_i = 0
        for in_sock_i in range(len(to_component.input_sockets)):
            # skip _variables Socket
            if from_component.output_sockets[out_sock_i].name == "_variables":
                out_sock_i += 1
            self.connect(from_component.output_sockets[out_sock_i], to_component.input_sockets[in_sock_i])
            out_sock_i += 1

    def OBSOLETE___getitem__(self, socket_name):
        """
        Use the []-lookup operator to get Sockets of this Component (in- and out-Socket names must be unique so it
        won't matter here, which type we lookup).

        Args:
            socket_name (str): The name of the Socket of this Component to return.

        Returns:
            Socket: The found Socket.

        Raises:
            KeyError: If Socket with the given name could not be found in this Component.
        """
        socket = self.get_socket_by_name(self, socket_name)
        if socket is None:
            raise KeyError("ERROR: Socket with name '{}' not found in Component '{}'!".format(socket_name, self.name))
        return socket

    def OBSOLETE___call__(self, *inputs):
        """
        FixMe: documentation.
        Args:
            *inputs ():

        Returns:

        """
        assert len(inputs) == len(self.input_sockets)

        # Create one DataOpRecord for each str input (an in-Socket's name).
        for i, in_op_spec in enumerate(util.force_list(inputs)):
            # Just a string: Create op_record with label so that only the Space from the Socket with that name
            # can pass through.
            if isinstance(in_op_spec, str):
                # Socket is the parent's component's Socket with that name.
                previous_socket = self.parent_component[in_op_spec]
                next_op_rec = DataOpRecord(group=self.op_group_id, filter_by_socket_name=previous_socket.name)
            # Already an op record.
            else:
                previous_socket = in_op_spec.socket
                assert previous_socket is not None
                next_op_rec = DataOpRecord(group=self.op_group_id)
                # Tell the in_op record where it has to push its primitive op to.
                in_op_spec.push_op_into_recs.add(next_op_rec)

            # Auto-create the Socket->Socket connection.
            self.connect(previous_socket, self.input_sockets[i])
            # Store the input_ops in our in-Sockets.
            self.input_sockets[i].op_records.add(next_op_rec)
            # Make the in-Socket aware that it's using individual op-connections.
            self.input_sockets[i].individual_in_connections = True

        # Create one output op_record for each return value of the combination of our graph_fn.
        out_ops = list()
        for out_sock in self.output_sockets:
            if out_sock.name == "_variables":
                continue
            out_op = DataOpRecord(group=self.op_group_id, socket=out_sock)
            # Add it to the respective out-Socket.
            out_sock.op_records.add(out_op)
            out_ops.append(out_op)

        self.op_group_id += 1

        return out_ops[0] if len(out_ops) == 1 else out_ops

    def OBSOLETE_get_socket(self, socket=None, type_=None, create_internal_if_not_found=False):
        """
        Returns a Component/Socket object pair given a specifier.

        Args:
            socket (Optional[Tuple[Component,str],str,Socket]):
                1) The name (str) of a local Socket.
                2) tuple: (Component, Socket-name OR Socket-object)
                3) Socket: An already given Socket -> return this Socket.
                4) None: Return the only Socket available on the given side.
            type_ (Optional[str]): Type of the Socket. If None, Socket could be either
                'in' or 'out'. This must be given if the only-Socket is wanted (socket is None).
            create_internal_if_not_found (bool): Whether to automatically create an internal Socket if the Socket
                cannot be found.

        Returns:
            Socket: The Socket found or automatically created.

        Raises:
            YARLError: If the Socket cannot be found and create_internal_if_not_found is False.
        """
        # Socket is given as a Socket object (simple pass-through).
        if isinstance(socket, Socket):
            return socket

        # Return the only Socket of this component on given side (type_ must be given in this case as 'in' or 'out').
        elif socket is None:
            assert type_ is not None, "ERROR: type_ needs to be specified if you want to get only-Socket!"
            if type_ == "out":
                socket_list = self.output_sockets
            else:
                socket_list = self.input_sockets

            assert len(socket_list) == 1, "ERROR: More than one {}-Socket! Cannot return only-Socket.".format(type_)
            return socket_list[0]

        # Possible constant-value for auto-generated internal Socket.
        constant_value = None

        # Socket is given as str: Try to look up socket by name.
        if isinstance(socket, str):
            socket_name = socket
            component = self
            socket_obj = self.get_socket_by_name(self, socket, type_=type_)

        # Socket is a constant value Socket defined by its value -> create this Socket with
        # a unique name and connect it to a constant value DataOp.
        elif isinstance(socket, (int, float, bool, np.ndarray)):
            socket_name = self.name + "-" + str(uuid.uuid1())
            component = self
            socket_obj = None
            constant_value = socket

        # Socket is given as component/sock-name OR component/sock-obj pair:
        # Could be ours, external, but also one of our sub-component's.
        else:
            assert len(socket) == 2,\
                "ERROR: Faulty socket spec! " \
                "External sockets need to be given as two item tuple: (Component, socket-name/obj)!"

            component = socket[0]  # type: Component
            if isinstance(component, str):
                component = self.sub_components.get(component)
                if component is None:
                    raise YARLError("ERROR: 1st item in tuple-Socket specifier is not the name of a sub-Component!".
                                    format(socket[0]))
            assert isinstance(component, Component), \
                "ERROR: First element in socket specifier is not of type Component, but of type: " \
                "'{}'!".format(type(component).__name__)
            socket_name = socket[1]
            # Socket is given explicitly -> use it.
            if isinstance(socket[1], Socket):
                socket_obj = socket[1]
            # Name is a string.
            else:
                socket_obj = self.get_socket_by_name(component, socket[1], type_=type_)

        # No Socket found.
        if socket_obj is None:
            # Create new internal one?
            if create_internal_if_not_found is True:
                self.add_socket(socket_name, value=constant_value, type_="in", internal=True)
                socket_obj = self.get_socket(socket_name)
            else:
                raise YARLError("ERROR: No {}Socket named '{}' found in {}!".
                                format("" if type_ is None else type_+"-", socket_name, component.name))

        return socket_obj

    @staticmethod
    def OBSOLETE_get_socket_by_name(component, name, type_=None):
        """
        Returns a Socket object of the given component by the name of the Socket. Or None if no Socket could be found.

        Args:
            component (Component): The Component object to search.
            name (str): The name of the Socket we are looking for.
            type_ (str): The type ("in" or "out") of the Socket we are looking for. Use None for any type.

        Returns:
            Optional[Socket]: The found Socket or None if no Socket by the given name was found in `component`.
        """
        if type_ != "in":
            socket_obj = next((x for x in component.output_sockets +
                               component.internal_sockets if x.name == name), None)
            if type_ is None and not socket_obj:
                socket_obj = next((x for x in component.input_sockets if x.name == name), None)
        else:
            socket_obj = next((x for x in component.input_sockets +
                               component.internal_sockets if x.name == name), None)
        return socket_obj

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

    def check_input_completeness(self):
        """
        Checks whether this Component is "input-complete" and stores the result in self.input_complete.
        Input-completeness is reached (only once and then it stays that way) if all in-Sockets to this component
        (whose name is not in `self.unconnected_sockets_in_meta_graph` have at least one op defined in
        their `ops` set.

        Returns:
            Optional[dict]: A space-dict if the Component is input-complete, None otherwise.
        """
        assert self.input_complete is False

        space_dict = dict()
        #self.input_complete = True
        #for in_socket in self.input_sockets:
        #    if in_socket.space is None and in_socket.name not in self.unconnected_sockets_in_meta_graph:
        #        self.input_complete = False
        #        return None
        #    else:
        #        space_dict[in_socket.name] = in_socket.space
        #return space_dict


        # Check, whether we are input-complete now.
        self.input_complete = True

        # Loop through all API methods.
        for method_name, api_method_rec in self.api_methods.items():
            # This API method must not be completed, ignore and don't add it to space_dict.
            if api_method_rec.must_be_complete is False:
                continue

            if api_method_rec.spaces is None:
                self.input_complete = False
                return None
            else:
                space_dict[method_name] = api_method_rec.spaces
            #space_dict[api_method_rec] = list()
            ## Loop through each of the APIMethodRecs' in op columns.
            #for in_op_col in api_method_rec.in_op_columns:
            #    # Loop through each op in the column.
            #    for op_rec in in_op_col:
            #        # Op not defined yet -> This column is not complete.
            #        if op_rec.op is None:
            #            break
            #    # Loop completed normally: All ops in this column are defined -> so far: input complete.
            #    else:
            #        break
            ## None of the op columns is complete -> We are not input complete.
            #else:
            #    self.input_complete = False
            #    return None
        return space_dict

    def _graph_fn__variables(self):
        """
        Outputs all of this Component's variables in a DataOpDict (out-Socket "_variables").

        This can be used e.g. to sync this Component's variables into another Component, which owns
        a Synchronizable() as a sub-component. The returns values of this graph_fn are then sent into
        the other Component's "_values" in-Socket for syncing.

        Returns:
            DataOpDict: Dict with keys=variable names and values=variable (SingleDataOp).
        """
        # Must use custom_scope_separator here b/c YARL doesn't allow Dict with '/'-chars in the keys.
        # '/' could collide with a FlattenedDataOp's keys and mess up the un-flatten process.
        variables_dict = self.get_variables(custom_scope_separator="-")
        return DataOpDict(variables_dict)

    def __str__(self):
        return "{}('{}' in={} out={})". \
            format(type(self).__name__, self.name, str(list(map(lambda s: s.name, self.input_sockets))),
                   str(list(map(lambda s: s.name, self.output_sockets))))
