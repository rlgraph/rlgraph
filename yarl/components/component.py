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

from yarl import YARLError, backend, Specifiable
from yarl.components.socket_and_computation import Socket, TfComputation
from yarl.utils import util
from yarl.spaces import Space

# some settings flags
EXPOSE_INS = 0x1  # whether to expose only in-Sockets (in calls to add_components)
EXPOSE_OUTS = 0x2  # whether to expose only out-Sockets (in calls to add_components)


class Component(Specifiable):
    """
    Base class for a graph component (such as a layer, an entire function approximator, a memory, an optimizer, etc..).

    A component can contain other components and/or its own computation (e.g. tf ops and tensors).
    A component's sub-components are connected to each other via in- and out-Sockets (similar to LEGO blocks
    and deepmind's sonnet).

    This base class implements the interface to add sub-components, create connections between
    different sub-components and between a sub-component and this one and between this component
     and an external component.

    A component also has a variable registry, saving the component's structure and variable-values to disk,
    adding computation to a component via tf.make_template calls.
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
        """

        # Scope if used to create scope hierarchies inside the Graph.
        self.scope = kwargs.get("scope", "")
        assert re.match(r'^[\w\-]*$', self.scope), \
            "ERROR: scope {} does not match scope-pattern! Needs to be \\w or '-'.".format(self.scope)
        # Names of sub-components that exist (parallelly) inside a containing component must be unique.
        self.name = kwargs.get("name", self.scope)  # if no name given, use scope
        self.device = kwargs.get("device")
        self.global_component = kwargs.get("global_component", False)

        # dict of sub-components that live inside this one (key=sub-component's scope)
        self.sub_components = dict()

        # Keep track of whether this Component has already been added to another Component and throw error
        # if this is done twice. Each Component should only be added once to some container Component for cleanlyness.
        self.has_been_added = False

        # This Component's in/out Sockets by name. As OrderedDicts for easier assignment to computation input-
        # parameters and return values.
        self.input_sockets = list()
        self.output_sockets = list()

        # All Variables that are held by this component by full name. This will always include sub-components'
        # variables.
        #self.variables = dict()

    def create_variables(self):
        """
        Should create all variables that are needed within this component,
        unless a variable is only needed inside a single computation method (_computation_-method), in which case,
        it should be created there.
        Variables must be created via the backend-agnostic self.get_variable-method.

        Note that for different scopes in which this component is being used, variables will not(!) be shared.
        """
        pass  # Children may overwrite this method.

    def get_variable(self, name="", shape=None, dtype="float", trainable=True, from_space=None):
        """
        Generates or returns a variable to use in the selected backend.

        Args:
            name (str): The name under which the variable is registered in this component.
            shape (Optional[tuple]): The shape of the variable (default: ()).
            dtype (Union[str,type]): The dtype (as string) of this variable.
            trainable (bool): Whether this variable should be trainable.
            from_space (Union[Space,None]): Whether to create this variable from a Space object
                (shape, dtype and trainable are not needed then).

        Returns:
            The actual variable (dependent on the backend).
        """
        # Called as getter.
        #if name in self.variables:
        #    return self.variables[name]

        # Called as setter.
        var = None

        if backend() == "tf":
            # TODO: need to discuss what other data we need per variable. Initializer, trainable, etc..
            if from_space:
                var = from_space.get_tensor_variable(name)
            else:
                if shape is None:
                    shape = tuple()
                var = tf.get_variable(name=name, shape=shape, dtype=util.dtype(dtype), trainable=trainable)
        # TODO: Register a new variable.
        #if var.name not in self.variables:
        #    self.variables[var.name] = var

        return var

    # Some helper functions (as special partials of existing methods).
    def define_inputs(self, *sockets, space=None):
        self.add_sockets(*sockets, type="in")
        if space is not None:
            for socket in sockets:
                self.connect(space, self.get_socket(socket).name)

    def define_outputs(self, *sockets, space=None):
        self.add_sockets(*sockets, type="out")
        if space is not None:
            for socket in sockets:
                self.connect(self.get_socket(socket).name, space)

    def add_sockets(self, *sockets, **kwargs):
        """
        Adds new input/output Sockets to this component.

        Args:
            *sockets (List[str]): List of names for the Sockets to be created.

        Keyword Args:
            type (str): Either "in" or "out". If no type given, try to infer type by the name.
                If name contains "input" -> type is "in", if name contains "output" -> type is "out".

        Raises:
            YARLError: If type is not given and cannot be inferred.
        """
        type_ = kwargs.get("type", kwargs.get("type_"))
        # Make sure we don't add two sockets with the same name.
        all_names = [sock.name for sock in self.input_sockets + self.output_sockets]

        for name in sockets:
            # Try to infer type by socket name (if name contains 'input' or 'output').
            if type_ is None:
                mo = re.search(r'([\d_]|\b)(in|out)put([\d_]|\b)', name)
                if mo:
                    type_ = mo.group(2)
                if type_ is None:
                    raise YARLError("ERROR: Cannot infer socket type (in/out) from socket name: '{}'!".format(name))

            if name in all_names:
                raise YARLError("ERROR: Socket of name '{}' already exists in component '{}'!".format(name, self.name))
            all_names.append(name)

            sock = Socket(name=name, scope=self.scope, type_=type_, device=self.device, global_=self.global_component)
            if type_ == "in":
                self.input_sockets.append(sock)
            else:
                self.output_sockets.append(sock)

    def add_computation(self, inputs, outputs, method_name=None):
        """
        Links a set (A) of sockets via a computation to a set (B) of other sockets via a computation function.
        Any socket in B is thus linked back to all sockets in A (requires all A sockets).

        Args:
            inputs (Union[str|List[str]]): List or single input Socket specifiers to link from.
            outputs (Union[str|List[str]]): List or single output Socket specifiers to link to.
            method_name (Union[str,None]): The (string) name of the template method to use for linking
                (without the _computation_ prefix). This method's signature and number of return values has to match the
                number of input- and output Sockets provided here. If None, use the only member method that starts with
                '_computation_' (error otherwise).

        Raises:
            YARLError: If method_name is None and not exactly one member method found that starts with `_computation_`.
        """
        inputs = util.force_list(inputs)
        outputs = util.force_list(outputs)

        if method_name is None:
            method_name = [m for m in dir(self) if (callable(self.__getattribute__(m)) and
                                                    re.match(r'^_computation_', m))]
            if len(method_name) != 1:
                raise YARLError("ERROR: Not exactly one method found that starts with '_computation_'! Cannot add "
                                "computation unless method_name is given explicitly.")
            method_name = re.sub(r'^_computation_', "", method_name[0])

        # TODO: Sanity check the tf methods signature and return type.
        # Compile a list of all input Sockets.
        input_sockets = [self.get_socket(s) for s in inputs]
        output_sockets = [self.get_socket(s) for s in outputs]
        # Add the computation record to all input and output sockets.
        computation = TfComputation(method_name, self, input_sockets, output_sockets)
        for input_socket in input_sockets:
            input_socket.connect_to(computation)
        for output_socket in output_sockets:
            output_socket.connect_from(computation)

    def add_component(self, component, expose=None):
        """
        Adds a single ModelComponent as sub-component to this one, thereby exposing certain Sockets of the
        sub-component to the interface of this component by creating the corresponding Sockets (and maybe renaming
        them depending on the specs in `expose`).

        Args:
            component (Component): The GrsphComponent object to add to this one.
            expose (any): Specifies, which of the Sockets of the added sub-component should be "forwarded" to this
                (containing) component via new Sockets of this component.
                For example: We are adding a sub-component with the Sockets: "input" and "output".
                expose="input": Only the "input" Socket is exposed into this component's interface (with name "input").
                expose={"input", "exposed-in"}: Only the "input" Socket is exposed into this component's interface as
                    new name "exposed-in".
                expose=("input", {"output": "exposed-out"}): The "input" Socket is exposed into this
                    component's interface. The output Socket as well, but under the name "exposed-out".
                expose=["input", "output"]: Both "input" and "output" Socket are exposed into this component's
                    interface (with their original names "input" and "output").
                expose=EXPOSE_INS: All in-Sockets will be exposed.
                expose=EXPOSE_OUTS: All out-Sockets will be exposed.
                expose=True: All sockets (in and out) will be exposed.
        """
        # Preprocess the expose spec.
        expose_spec = dict()
        if expose is not None:
            # More than one socket needs to be exposed.
            if isinstance(expose, (list, tuple)):
                for e in expose:
                    if isinstance(e, str):
                        expose_spec[e] = e  # leave name
                    elif isinstance(e, (tuple, list)):
                        expose_spec[e[0]] = e[1]  # change name from 1st element to 2nd element in tuple/list
                    elif isinstance(e, dict):
                        for old, new in e.items():
                            expose_spec[old] = new  # change name from dict-key to dict-value
            else:
                # Expose all Sockets if expose=True|EXPOSE_INS|EXPOSE_OUTS.
                expose_list = list()
                if expose == EXPOSE_INS or expose is True:
                    expose_list.extend(component.input_sockets)
                if expose == EXPOSE_OUTS or expose is True:
                    expose_list.extend(component.output_sockets)
                if len(expose_list) > 0:
                    for sock in expose_list:
                        expose_spec[sock.name] = sock.name
                # Single socket (given as string) needs to be exposed (and keep its name).
                else:
                    expose_spec[expose] = expose  # leave name

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
        for socket_name, exposed_name in expose_spec.items():
            socket = self.get_socket_by_name(component, socket_name)  # type: Socket
            if socket is None:
                raise YARLError("ERROR: Could not find Socket '{}' in input/output sockets of component '{}'!".
                                format(socket_name, self.name))
            new_socket = self.get_socket_by_name(self, exposed_name)
            # Doesn't exist yet -> add it.
            if new_socket is None:
                self.add_sockets(exposed_name, type=socket.type)
            # Connect the two Sockets.
            self.connect(exposed_name, [component, socket_name])

    def add_components(self, *components, expose=None):
        """
        Adds sub-components to this one without connecting them with each other.

        Args:
            *components (Component): The list of ModelComponent objects to be added into this one.
            expose (Union[dict,tuple,str]): Expose-spec for the component(s) to be passed to self.add_component().
                If more than one sub-components are added in the call and expose is a dict, lookup each component's
                name in that dict and pass the found value to self.add_component. If expose is not a dict, pass it
                as-is for each of the added sub-components.
        """
        for c in components:
            self.add_component(c, expose.get(c.name) if isinstance(expose, dict) else expose)

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
            The copied component object.
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
        """
        # Connect a Space (other must be Socket).
        # Also, there are certain restrictions for the Socket's type.
        if isinstance(from_, Space):
            to_socket_obj = self.get_socket(to_)
            assert to_socket_obj.type == "in", "ERROR: Cannot connect a Space to an 'out'-Socket!"
            if not disconnect:
                to_socket_obj.connect_from(from_)
            else:
                to_socket_obj.disconnect_from(from_)
            return
        elif isinstance(to_, Space):
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

        # Connect the two Sockets in both ways.
        if not disconnect:
            from_socket_obj.connect_to(to_socket_obj)
            to_socket_obj.connect_from(from_socket_obj)
        else:
            from_socket_obj.disconnect_to(to_socket_obj)
            to_socket_obj.disconnect_from(from_socket_obj)

    def get_socket(self, socket=None, type_=None, return_component=False):
        """
        Returns a Component/Socket object pair given a specifier.
        Does all error checking.

        Args:
            socket (Optional[Tuple[Component,str],str,Socket]):
                1) The name (str) of a local Socket (including sub-component's ones using the
                    "sub-comp-nam/socket-name"-notation)
                2) tuple: (Component, Socket-name OR Socket-object)
                3) Socket: An already given Socket -> return this Socket (throw error if return_component is True).
                4) None: Return the only Socket available on the given side.
            type_ (Union[None,str]): Type of the Socket. If None, Socket could be either
                'in' or 'out'. This must be given if the only-Socket is wanted (socket is None).
            return_component (bool): Whether also to return the Socket's component as a second element in a tuple.

        Returns: Either only the Socket found OR a tuple of the form:
            (retrieved Socket object, component that the retrieved socket belongs to).
        """

        # Return the only Socket of this component on given side (type_ must be given in this case as 'in' or 'out').
        if socket is None:
            assert type_ is not None, "ERROR: type_ needs to be specified if you want to get only-Socket!"
            if type_ == "out":
                list_ = self.output_sockets
            else:
                list_ = self.input_sockets

            assert len(list_) == 1, "ERROR: More than one {}-Socket! Cannot return only-Socket.".format(type_)
            return list_[0]

        # Socket is given as string-only: Try to look up socket via this string identifier.
        if isinstance(socket, str):
            socket_name = socket
            mo = re.match(r'^([\w\-]*)\/(.+)$', socket)
            if mo:
                assert mo.group(1) in self.sub_components, "ERROR: Sub-component '' does not exist in this component!".\
                    format(mo.group(1))
                component = self.sub_components[mo.group(1)]
                socket = mo.group(2)
            else:
                component = self
            socket_obj = self.get_socket_by_name(component, socket, type_=type_)
        # Socket is given as a Socket object (simple pass-through).
        elif isinstance(socket, Socket):
            if return_component is True:
                raise YARLError("ERROR: Cannot pass through Socket if parmaater `socket` is given as a Socket object "
                                "and return_component is True!")
            return socket
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

        if socket_obj is None:
            raise YARLError("ERROR: No '{}'-socket named '{}' found in {}!". \
                            format("??" if type_ is None else type_, socket_name, component.name))

        return socket_obj if not return_component else (component, socket_obj)

    def get_input(self, socket=None):
        """
        Helper method to retrieve one of our own in-Sockets by name (None for the only in-Socket
        there is).

        Args:
            socket (Optional[str]): The name of the in-Socket to retrieve.

        Returns:
            The found in-Socket.
        """
        return self.get_socket(socket, type_="in", return_component=True)

    def get_output(self, socket=None):
        """
        Helper method to retrieve one of our own out-Sockets by name (None for the only out-Socket
        there is).

        Args:
            socket (Optional[str]): The name of the out-Socket to retrieve.

        Returns:
            The found out-Socket.
        """
        return self.get_socket(socket, type_="out", return_component=True)

    @staticmethod
    def get_socket_by_name(component, name, type_=None):
        if type_ != "in":
            socket_obj = next((x for x in component.output_sockets if x.name == name), None)
            if type_ is None and not socket_obj:
                socket_obj = next((x for x in component.input_sockets if x.name == name), None)
        else:
            socket_obj = next((x for x in component.input_sockets if x.name == name), None)
        return socket_obj

