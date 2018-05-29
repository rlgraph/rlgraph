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

import tensorflow as tf
import re
import copy
from collections import Hashable
import numpy as np
import uuid

from yarl import YARLError, backend, Specifiable
from yarl.utils.ops import SingleDataOp
from yarl.components.socket_and_graph_fn import Socket, GraphFunction
from yarl.utils import util
from yarl.spaces import Space

# some settings flags
CONNECT_INS = 0x1  # whether to expose only in-Sockets (in calls to add_components)
CONNECT_OUTS = 0x2  # whether to expose only out-Sockets (in calls to add_components)


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
    def __init__(self, *args, **kwargs):
        """
        Args:
            args (any): For subclasses to use.

        Keyword Args:
            name (str): The name of this Component. Names of sub-components within a containing component
                must be unique. Names are used to label exposed Sockets of the containing component.
                If name is empty, use scope as name (as last resort).
            scope (str): The scope of this Component for naming variables in the Graph.
            device (str): Device this component will be assigned to. If None, defaults to CPU.
            global_component (bool): In distributed mode, this flag indicates if the component is part of the
                shared global model or local to the worker. Defaults to False and will be ignored if set to
                True in non-distributed mode.
            graph_fn_settings (dict): Dict with possible general GraphFunction settings that should be applied to
                all of this Component's GraphFunctions. See `self.add_graph_fn` and GraphFunction's c'tor for more
                details.
        """

        # Scope if used to create scope hierarchies inside the Graph.
        self.scope = kwargs.pop("scope", "")
        assert re.match(r'^[\w\-]*$', self.scope), \
            "ERROR: scope {} does not match scope-pattern! Needs to be \\w or '-'.".format(self.scope)
        # Names of sub-components that exist (parallelly) inside a containing component must be unique.
        self.name = kwargs.pop("name", self.scope)  # if no name given, use scope
        self.device = kwargs.pop("device", None)
        self.global_component = kwargs.pop("global_component", False)

        # TODO: get rid of this again and add the 4 settings as single parameters to this c'tor.
        self.graph_fn_settings = kwargs.pop("graph_fn_settings", {})

        assert not kwargs, "ERROR: kwargs ({}) still contains items!".format(kwargs)

        # Dict of sub-components that live inside this one (key=sub-component's scope).
        self.sub_components = dict()

        # Keep track of whether this Component has already been added to another Component and throw error
        # if this is done twice. Each Component should only be added once to some container Component for cleanlyness.
        self.has_been_added = False

        # This Component's Socket objects by functionality.
        self.input_sockets = list()  # the exposed in-Sockets
        self.output_sockets = list()  # the exposed out-Sockets
        self.internal_sockets = list()  # internal (non-exposed) in/out-Sockets (e.g. used to connect 2 GraphFunctions)

        # Whether we know already all our in-Sockets' Spaces.
        # Only then can we create our variables. Model will do this.
        self.input_complete = False

        # Collect Sockets that we need to built later, directly after this component
        # is input-complete. This input-completeness may happen at another Socket and thus some Sockets
        # need to be built later.
        self.sockets_to_do_later = list()

        # Contains our GraphFunctions that have no in-Sockets and thus will not be included in the forward search.
        # Thus, these need to be treated separately.
        self.no_input_entry_points = list()

        # All Variables that are held by this component (and its sub-components?) by name.
        # key=full-scope variable name
        # value=the actual variable
        self.variables = dict()

    def create_variables(self, input_spaces):
        """
        Should create all variables that are needed within this component,
        unless a variable is only needed inside a single _graph_fn-method, in which case,
        it should be created there.
        Variables must be created via the backend-agnostic self.get_variable-method.

        Note that for different scopes in which this component is being used, variables will not(!) be shared.

        Args:
            input_spaces (Dict[str,Space]): A convenience dict with Space/shape information.
                keys=in-Socket name (str); values=the associated Space
        """
        print("Creating Variables for {}".format(self.name))  # Children may overwrite this method.

    def get_variable(self, name="", shape=None, dtype="float", initializer=None, trainable=True,
                     from_space=None, add_batch_rank=False, flatten=False):
        """
        Generates or returns a variable to use in the selected backend.
        The generated variable is automatically registered in this component's variable registry under
        its name.

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
                var = from_space.get_tensor_variable(name=name, add_batch_rank=add_batch_rank, trainable=trainable)
        # Direct variable creation (using the backend).
        elif backend() == "tf":
            # Provide a shape, if initializer is not given or it is an actual Initializer object (rather than an array
            # of fixed values, for which we then don't need a shape as it comes with one).
            if initializer is None or isinstance(initializer, tf.keras.initializers.Initializer):
                shape = tuple((() if add_batch_rank is False else
                                 (None,) if add_batch_rank is True else (add_batch_rank,)) + (shape or ()))
            else:
                shape = None

            var = tf.get_variable(
                name=name,
                shape=shape,
                dtype=util.dtype(dtype),
                initializer=initializer,
                trainable=trainable
            )

        # Registers the new variable in this Component.
        self.variables[name] = var

        return var

    def get_variables(self, *names,  **kwargs):
        """
        Utility method to get one or more component variable(s) by name(s).

        Args:
            names (List[str]): Lookup name strings for variables. None for all.

        Keyword Args:
            collections (set): A set of collections to which the variables have to belong in order to be returned here.
                Default: tf.GraphKeys.TRAINABLE_VARIABLES

        Returns:
            dict: A dict mapping variable names to their backend variables.
        """
        collection = kwargs.pop("collection", None)
        assert not kwargs, "{}".format(kwargs)

        if len(names) == 1 and isinstance(names[0], list):
            names = names[0]
        names = util.force_list(names)
        # Return all variables of this Component (for some collection).
        if len(names) == 0:
            collection_variables = tf.get_collection(collection)
            return {v.name: v for v in collection_variables if v.name in self.variables}

        # Return only variables of this Component by name.
        return {n: self.variables[n] for n in names if n in self.variables}

    def define_inputs(self, *sockets, **kwargs):
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

    def define_outputs(self, *sockets):
        """
        Calls add_sockets with type="out".

        Args:
            *sockets (List[Socket]): The list of out-Sockets to add.
        """
        # TODO: Add out-Socket to Space connection for reverse Space inferral?
        # TODO: E.g.: NN-fc-out layer -> Action Space (then the out layer would know how many units it needs).
        self.add_sockets(*sockets, type="out")

    def add_socket(self, name, value=None, **kwargs):
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

    def add_sockets(self, *socket_names, **kwargs):
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

    def add_graph_fn(self, inputs, outputs, method=None,
                        flatten_ops=None, split_ops=None,
                        add_auto_key_as_first_param=None, unflatten_ops=None):
        """
        Links a set (A) of sockets via a graph_fn to a set (B) of other sockets via a graph_fn function.
        Any socket in B is thus linked back to all sockets in A (requires all A sockets).

        Args:
            inputs (Optional[str,List[str]]): List or single input Socket specifiers to link from.
            outputs (Optional[str,List[str]]): List or single output Socket specifiers to link to.
            method (Optional[str,callable]): The name of the template method to use for linking
                (without the _graph_ prefix) or the method itself (callable).
                The `method`'s signature and number of return values has to match the
                number of input- and output Sockets provided here.
                If None, use the only member method that starts with '_graph_' (error otherwise).
            flatten_ops (Optional[bool,Set[str]]): Passed to GraphFunction's c'tor. See GraphFunction
                for details. Overwrites this Component's `self.graph_fn_settings`.
            split_ops (Optional[bool,Set[str]]): Passed to GraphFunction's c'tor. See GraphFunction
                for details. Overwrites this Component's `self.graph_fn_settings`.
            add_auto_key_as_first_param (Optional[bool]): Passed to GraphFunction's c'tor. See GraphFunction for details.
                Overwrites this Component's `self.graph_fn_settings`.
            unflatten_ops (Optional[bool]): Passed to GraphFunction's c'tor. See GraphFunction for details.
                Overwrites this Component's `self.graph_fn_settings`.

        Raises:
            YARLError: If method_name is None and not exactly one member method found that starts with `_graph_`.
        """
        inputs = util.force_list(inputs)
        outputs = util.force_list(outputs)

        if method is None:
            method = [m for m in dir(self) if (callable(self.__getattribute__(m)) and re.match(r'^_graph_', m))]
            if len(method) != 1:
                raise YARLError("ERROR: Not exactly one method found in {} that starts with '_graph_'! "
                                "Cannot add graph_fn unless method_name is given explicitly.".format(self.name))
            method = re.sub(r'^_graph_', "", method[0])

        # TODO: Make it possible to call other graph_fns within a graph_fn and to call a sub-component's graph_fn within a graph_fn.
        # TODO: Sanity check the tf methods signature (and return values from docstring?).
        # Compile a list of all needed Sockets and create internal ones if they do not exist yet.
        # External Sockets (in/out) must exist already or we will get an error.
        input_sockets = [self.get_socket(s, create_internal_if_not_found=True) for s in inputs]
        output_sockets = [self.get_socket(s, create_internal_if_not_found=True) for s in outputs]
        # Add the graph_fn record to all input and output sockets.
        # Fetch default params.
        if flatten_ops is None:
            flatten_ops = self.graph_fn_settings.get("flatten_ops", True)
        if split_ops is None:
            split_ops = self.graph_fn_settings.get("split_ops", False)
        if add_auto_key_as_first_param is None:
            add_auto_key_as_first_param = self.graph_fn_settings.get("add_auto_key_as_first_param", False)
        if unflatten_ops is None:
            unflatten_ops = self.graph_fn_settings.get("unflatten_ops", True)

        graph_fn = GraphFunction(
            method=method,
            component=self,
            input_sockets=input_sockets,
            output_sockets=output_sockets,
            flatten_ops=flatten_ops,
            split_ops=split_ops,
            add_auto_key_as_first_param=add_auto_key_as_first_param,
            unflatten_ops=unflatten_ops
        )
        # Connect the graph_fn to all the given Sockets.
        for input_socket in input_sockets:
            input_socket.connect_to(graph_fn)
        for output_socket in output_sockets:
            output_socket.connect_from(graph_fn)

        # If the GraphFunction has no inputs or only constant value inputs, we need to build it from no-input.
        if len(input_sockets) == 0:
            self.no_input_entry_points.append(graph_fn)

    def add_component(self, component, connect=None):
        """
        Adds a single Component as a sub-component to this one, thereby connecting certain Sockets of the
        sub-component to the Sockets of this component.
        - If to-be-connected Sockets of this Component do not exist yet, they will be created automatically
            (and maybe renamed depending on the `connect` spec).
        - Alternatively, in-Sockets of the sub-component may be added directly to constant values. In this case,
            no new Sockets need to be generated.

        Args:
            component (Component): The Component object to add to this one.
            connect (any): Specifies, which of the Sockets of the added sub-component should be connected to this
                (containing) component via connecting to (sometimes new) Sockets of this component.
                For example: We are adding a sub-component with the Sockets: "input" and "output".
                connect="input": Only the "input" Socket is connected to this component's "input" Socket.
                connect={"input", "exposed-in"}: Only the "input" Socket is connected to this component's Socket named
                    "exposed-in".
                connect=("input", {"output": "exposed-out"}): The "input" Socket is connected to this
                    component's "input". The output Socket to a Socket named "exposed-out".
                connect=["input", "output"]: Both "input" and "output" Sockets are connected into this component's
                    interface (with their original names "input" and "output").
                connect=CONNECT_INS: All in-Sockets of `component` will be connected.
                connect=CONNECT_OUTS: All out-Sockets of `component` will be connected.
                connect=True: All sockets of `component` (in and out) will be connected.
                connect={"input": np.array([[1, 2], [3, 4]])}: Connects the "input" Socket of the sub-component
                    to a constant value DataOp with the given numpy value. This also works for python primitives
                    (float, int, and bool).
        """
        # Preprocess the connect spec.
        connect_spec = dict()
        if connect is not None:
            # More than one socket needs to be exposed.
            if isinstance(connect, (list, tuple)):
                for e in connect:
                    if isinstance(e, str):
                        connect_spec[e] = e  # leave name
                    elif isinstance(e, (tuple, list)):
                        connect_spec[e[0]] = e[1]  # change name from 1st element to 2nd element in tuple/list
                    elif isinstance(e, dict):
                        for old, new in e.items():
                            connect_spec[old] = new  # change name from dict-key to dict-value
            elif isinstance(connect, dict):
                for old, new in connect.items():
                    connect_spec[old] = new  # change name from dict-key to dict-value
            else:
                # Expose all Sockets if connect=True|CONNECT_INS|CONNECT_OUTS.
                connect_list = list()
                if connect == CONNECT_INS or connect is True:
                    connect_list.extend(component.input_sockets)
                if connect == CONNECT_OUTS or connect is True:
                    connect_list.extend(component.output_sockets)
                if len(connect_list) > 0:
                    for sock in connect_list:
                        connect_spec[sock.name] = sock.name
                # Single socket (given as string) needs to be exposed (and keep its name).
                else:
                    connect_spec[connect] = connect  # leave name

        # Make sure no two components with the same name are added to this one (own scope doesn't matter).
        if component.name in self.sub_components:
            raise YARLError("ERROR: Sub-Component with name '{}' already exists in this one!".format(component.name))
        # Make sure each Component can only be added once to a parent/container Component.
        elif component.has_been_added is True:
            raise YARLError("ERROR: Sub-Component with name '{}' has already been added once to a container Component! "
                            "Each Component can only be added once to a parent.".format(component.name))
        component.has_been_added = True
        self.sub_components[component.name] = component

        # Expose all Sockets in exposed_spec (create and connect them correctly).
        for socket_name, exposed_name_or_value in connect_spec.items():
            socket = self.get_socket_by_name(component, socket_name)  # type: Socket
            if socket is None:
                raise YARLError("ERROR: Could not find Socket '{}' in input/output sockets of component '{}'!".
                                format(socket_name, self.name))
            new_socket = self.get_socket_by_name(self, exposed_name_or_value)
            # Doesn't exist yet -> add it.
            if new_socket is None:
                # A constant value Socket -> create artificial name and connect directly with constant op (on python
                # side, not a constant in the graph).
                if not isinstance(exposed_name_or_value, str):
                    exposed_name_or_value = SingleDataOp(constant_value=exposed_name_or_value)
                else:
                    self.add_socket(exposed_name_or_value, type=socket.type)

            # Connect the two Sockets.
            if socket.type == "in":
                self.connect(exposed_name_or_value, [component, socket])
            else:
                self.connect([component, socket], exposed_name_or_value)

    def add_components(self, *components, **kwargs):
        """
        Adds sub-components to this one without connecting them with each other.

        Args:
            *components (Component): The list of ModelComponent objects to be added into this one.

        Keyword Args:
            connect (Union[dict,tuple,str]): Connection-spec for the component(s) to be passed to self.add_component().
                If more than one sub-components are added in the call and `connect` is a dict, lookup each component's
                name in that dict and pass the found value to self.add_component. If `connect` is not a dict, pass it
                as-is for each of the added sub-components.
        """
        connect = kwargs.pop("connect", None)
        assert not kwargs

        for c in components:
            connect_ = None
            # If `connect` is a dict, it may be a dict with keys=component name but could also be a dict
            # with keys=Sockets to be renamed. Figure this out here and pass it to `self.add_component` accordingly.
            if isinstance(connect, dict):
                connect_ = connect.get(c.name)
            self.add_component(c, (connect_ or connect))

    def copy(self, name=None, scope=None):
        """
        Copies this component and returns a new component with possibly another name and another scope.
        The new component has its own variables (they are not shared with the variables of this component)
        and is initially not connected to any other component. However, the Sockets of this component and their names
        are being copied (but without their connections).

        Args:
            name (str): The name of the new component. If None, use the value of scope.
            scope (str): The scope of the new component. If None, use the same scope as this component.

        Returns:
            Component: The copied component object.
        """
        if scope is None:
            scope = self.scope
        if name is None:
            name = scope

        # Simply deepcopy self and change name and scope.
        new_component = copy.deepcopy(self)
        new_component.name = name
        new_component.scope = scope

        # Then cut all the new_component's outside connections (no need to worry about the other side as
        # they were not notified of the copied Sockets).
        for socket in new_component.input_sockets:
            socket.incoming_connections = list()
        for socket in new_component.output_sockets:
            socket.outgoing_connections = list()

        return new_component

    def connect(self, from_, to_):
        """
        Makes a connection between:
        - a Socket (from_) and another Socket (to_).
        - a Socket and a Space (or the other way around).
        from_ and to_ may not be Socket objects but Socket-specifiers. See self.get_socket for details on possible ways
        to specify a socket (by string, component-name, etc..).
        Also, from_ and/or to_ may be Component objects. In that case, all out connections of from_ will be
        connected to the respective in connections of to_.

        Args:
            from_ (any): The specifier of the connector (e.g. incoming Socket).
            to_ (any): The specifier of the connectee (e.g. another Socket).
        """
        self._connect_disconnect(from_, to_)

    def disconnect(self, from_, to_):
        """
        Removes a connection between:
        - a Socket (from_) and another Socket (to_).
        - a Socket and a Space (or the other way around).
        from_ and to_ may not be Socket objects but Socket-specifiers. See self.get_socket for details on possible ways
        to specify a socket (by string, component-name, etc..).
        Also, from_ and/or to_ may be Component objects. In that case, all out connections of from_ to all in
        connections of to_ are cut.

        Args:
            from_ (any): The specifier of the connector (e.g. incoming Socket).
            to_ (any): The specifier of the connectee (e.g. another Socket).
        """
        self._connect_disconnect(from_, to_, disconnect=True)

    def _connect_disconnect(self, from_, to_, disconnect=False):
        """
        Actual private implementer for `connect` and `disconnect`.

        Args:
            from_ (any): The specifier of the connector (e.g. incoming Socket, an incoming Space).
            to_ (any): The specifier of the connectee (e.g. another Socket).
            disconnect (bool): Only used internally. Whether to actually disconnect (instead of connect).

        Raises:
            YARLError: If one tries to connect two Sockets of the same Component.
        """
        # Connect a Space (other must be Socket).
        # Also, there are certain restrictions for the Socket's type.
        if isinstance(from_, (Space, dict, SingleDataOp)) or \
                (not isinstance(from_, str) and isinstance(from_, Hashable) and from_ in Space.__lookup_classes__):
            from_ = from_ if isinstance(from_, (Space, SingleDataOp)) else Space.from_spec(from_)
            to_socket_obj = self.get_socket(to_)
            assert to_socket_obj.type == "in", "ERROR: Cannot connect a Space to an 'out'-Socket!"
            if not disconnect:
                to_socket_obj.connect_from(from_)
            else:
                to_socket_obj.disconnect_from(from_)
            return
        elif isinstance(to_, (Space, dict)) or \
                (not isinstance(to_, str) and isinstance(to_, Hashable) and to_ in Space.__lookup_classes__):
            to_ = to_ if isinstance(to_, Space) else Space.from_spec(to_)
            from_socket_obj = self.get_socket(from_)
            assert from_socket_obj.type == "out", "ERROR: Cannot connect an 'in'-Socket to a Space!"
            if not disconnect:
                from_socket_obj.connect_to(to_)
            else:
                from_socket_obj.disconnect_to(to_)
            return

        # Get only-out Socket.
        if isinstance(from_, Component):
            from_socket_obj = from_.get_socket(type_="out")
        # Regular Socket->Socket connection.
        else:
            from_socket_obj = self.get_socket(from_)

        # Get only-in Socket.
        if isinstance(to_, Component):
            to_socket_obj = to_.get_socket(type_="in")
        else:
            to_socket_obj = self.get_socket(to_)

        # (Dis)connect the two Sockets in both ways.
        if disconnect:
            # Sanity check that we are not connecting two Sockets of the same Component.
            if from_socket_obj.component is to_socket_obj.component:
                raise YARLError("ERROR: Cannot connect two Sockets that belong to the same Component!")

            from_socket_obj.disconnect_to(to_socket_obj)
            to_socket_obj.disconnect_from(from_socket_obj)
        else:
            from_socket_obj.connect_to(to_socket_obj)
            to_socket_obj.connect_from(from_socket_obj)

    def get_socket(self, socket=None, type_=None, create_internal_if_not_found=False):
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
                cannot be found (in)

        Returns:
            Socket: Only the Socket found.
            Tuple[Socket,Component]: Retrieved Socket object, Component that the retrieved Socket belongs to.

        Raises:
            YARLError: If the Socket cannot be found and create_internal_if_not_found is False.
        """
        constant_dataop = None  # possible constant value for a new Socket

        # Return the only Socket of this component on given side (type_ must be given in this case as 'in' or 'out').
        if socket is None:
            assert type_ is not None, "ERROR: type_ needs to be specified if you want to get only-Socket!"
            if type_ == "out":
                list_ = self.output_sockets
            else:
                list_ = self.input_sockets

            assert len(list_) == 1, "ERROR: More than one {}-Socket! Cannot return only-Socket.".format(type_)
            return list_[0]
        # Socket is given as str: Try to look up socket by name.
        elif isinstance(socket, str):
            socket_name = socket
            component = self
            socket_obj = self.get_socket_by_name(self, socket, type_=type_)
        # Socket is given as a Socket object (simple pass-through).
        elif isinstance(socket, Socket):
            return socket
        # Socket is a constant value Socket defined by its value -> create this Socket with
        # a unique name and connect it to a constant value DataOp.
        elif isinstance(socket, (int, float, bool, np.ndarray)):
            socket_name = self.name + "-" + str(uuid.uuid1())
            component = self
            socket_obj = None
            constant_dataop = SingleDataOp(constant_value=socket)
        # Socket is given as component/sock-name OR component/sock-obj pair:
        # Could be ours, external, but also one of our sub-component's.
        else:
            assert len(socket) == 2,\
                "ERROR: Faulty socket spec! " \
                "External sockets need to be given as two item tuple: (Component, socket-name/obj)!"
            assert isinstance(socket[0], Component),\
                "ERROR: First element in socket specifier is not of type Component, but of type: '{}'!".\
                    format(type(socket[0]).__name__)
            component = socket[0]  # type: Component
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
                self.add_socket(socket_name, value=constant_dataop, type_="in", internal=True)
                socket_obj = self.get_socket(socket_name)
            # Error.
            else:
                raise YARLError("ERROR: No '{}'-socket named '{}' found in {}!". \
                                format("??" if type_ is None else type_, socket_name, component.name))

        return socket_obj

    def get_input(self, socket=None):
        """
        Helper method to retrieve one of our own in-Sockets by name (None for the only in-Socket
        there is).

        Args:
            socket (Optional[str]): The name of the in-Socket to retrieve.

        Returns:
            Socket: The found in-Socket.
        """
        return self.get_socket(socket, type_="in")

    def get_output(self, socket=None):
        """
        Helper method to retrieve one of our own out-Sockets by name (None for the only out-Socket
        there is).

        Args:
            socket (Optional[str]): The name of the out-Socket to retrieve.

        Returns:
            Socket: The found out-Socket.
        """
        return self.get_socket(socket, type_="out")

    @staticmethod
    def get_socket_by_name(component, name, type_=None):
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
        if backend() == "tf":
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
        if backend() == "tf":
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
        if backend() == "tf":
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
        have at least one op defined in their Socket.ops set.
        """
        if not self.input_complete:
            # Check whether we now have all in-Sockets with a Space-information.
            self.input_complete = True
            for in_sock in self.input_sockets:
                if in_sock.space is None:
                    self.input_complete = False

    def __str__(self):
        return "{}('{}' in=[{}] out=[{}])". \
            format(type(self).__name__, self.name, str(list(map(str, self.input_sockets))),
                   str(list(map(str, self.output_sockets))))

