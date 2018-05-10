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

import itertools
from tensorflow.contrib import autograph

from yarl.spaces import Space


class Socket(object):
    """
    A Socket object describes a connection to other Sockets, Computations, or Spaces inside and between ModelComponents.
    One Socket either carries:

    - a single op (e.g. some tensor)
    - a tuple of ops (nested also supported)
    - a dict of ops (nested also supported)

    Also, each of the above possibilities can have many parallel solutions. These splits happen e.g. if two Sockets
    connect to the same target Socket. In this case, the target Socket's inputs are treated as possible alternatives
    and the Socket then implicitly produces two outputs that it further passes on to the next Sockets/Computations.

    When connected to a computation object, a Socket always represents one of the input parameters to the computation
    method. Also, each returned value of a computation method corresponds to one Socket.
    """
    def __init__(self, name="", scope="", type_="in", device=None, global_=False):
        """
        Args:
            name (str): The name of this Socket (as it will show in the final call interface).
            scope (str): The scope of this Socket (same as the owning Component's scope).
            type_ (str): The Socket type: "in" or "out".
            device (str): Device this Socket's component will be assigned to. If None, defaults to CPU.
            global_ (bool): In distributed mode, this flag indicates if the Socket's component is part of the
                shared global model or local to the worker. Defaults to False and will be ignored if set to
                True in non-distributed mode.
        """
        # The name of this Socket.
        self.name = name
        # The scope of this socket (important for computations and variable creation).
        self.scope = scope
        # "in" or "out"
        self.type = type_
        # The device to use when placing ops that come after this Socket into the Graph.
        self.device = device
        self.global_ = global_

        # Which other socket(s), space(s), computation(s) are we connected to on the incoming and outgoing side?
        # - Records in these lists have a parallel relationship (they are all alternatives to each other).
        # - Each record is either:
        #   - another Socket
        #   - a Space
        #   - a dict of: {computation-method, slot for this socket, [required input sockets], [output sockets]}
        self.incoming_connections = list()
        self.outgoing_connections = list()

        # The set of (alternative) tf op(s) that this socket carries. Populated at build time.
        self.ops = set()

    # A+B -> comp -> C+D -> C -> E (which also gets an alternative input from G) AND D -> F
    # A arrives -> wait (no B yet)
    # b arrives -> comp(A,b) -> C1+D1(given A+b)
    # a arrives -> comp(a,b) -> C2+D2(given a+b)
    # B arrives -> comp(a,B) -> C3+D3(given a+B) AND comp(A,B) -> C4+D4(given A+B)
    # now we have C with ops: {opC1: [A,b], opC2: [a,b], opC3: [a,B], opC4:[A,B]}
    # same for D
    # E: incoming_connections: [C, G] -> pass on ops from C to E (just copy!) and concat(!) with all ops from G
    # expose E as interface and make it "callable", then pull ops depending on input given (e.g. [a,b] -> pull opC2).
    def connect_to(self, to_):
        if to_ not in self.outgoing_connections:
            self.outgoing_connections.append(to_)

    def disconnect_to(self, to_):
        if to_ in self.outgoing_connections:
            self.outgoing_connections.remove(to_)

    def connect_from(self, from_):
        if from_ not in self.incoming_connections:
            self.incoming_connections.append(from_)

    def disconnect_from(self, from_):
        if from_ in self.incoming_connections:
            self.incoming_connections.remove(from_)

    def update_from_input(self, incoming, op_registry, socket_registry, slot=None):
        """
        Updates this socket based on an incoming connection (from a Space or Computation or another Socket).

        Args:
            incoming (Union[Space,Computation,Socket]): The incoming item.
            op_registry (dict): Dict that keeps track of which ops require which other ops to be calculated.
            socket_registry (dict): Dict that keeps track of which very in-Socket needs which ops (placeholders/feeds).
            slot (int): If incoming is a Computation, which output slot does this Socket connect to?
        """
        # Space: generate backend-ops.
        if isinstance(incoming, Space):
            # TODO: Add batch dimension-option.
            # NOTE: op could be dict or tuple as well.
            op = incoming.get_tensor_variable(name=self.name, is_input_feed=True)
            # Add new op to our list of (alternative) ops.
            self.ops.add(op)
            # Keep track of which Spaces can go (alternatively) into this Socket.
            socket_registry[self] = {op} if self not in socket_registry else (socket_registry[self] | {op})
            # Remember, that this op goes into a Socket at the very beginning of the Graph (e.g. a tf.placeholder).
            op_registry[op] = {self}
        # Computation: Connect this Socket to the nth op coming out of the Computation function.
        elif isinstance(incoming, TfComputation):
            assert isinstance(slot, int) and slot >= 0, "ERROR: If incoming is a Computation, slot must be set and >=0!"
            # Add every nth op from the output of the completed computations to this Socket's set of ops.
            nth_computed_ops = [outputs[slot] for outputs in incoming.processed_ops]
            self.ops.update(nth_computed_ops)
        # Incoming is another Socket -> Simply add ops to this one.
        else:
            assert isinstance(incoming, Socket), "ERROR: incoming must be Socket!"
            self.ops.update(incoming.ops)

    def __str__(self):
        return "{}-Socket('{}/{}'{})".format(self.type, self.scope, self.name,
                                             " dev='{}'".format(self.device) if self.device else "")


class Computation(object):
    """
    Class describing a computation happening inside a _computation-func inside a Component.
    Implements the update_from_input method which checks whether all necessary inputs to a computation function
    are given and - if yes - starts producing output ops from these inputs and the computation to be passed
    on to the outgoing Sockets.
    """
    def __init__(self, name, component, input_sockets, output_sockets):
        """
        Args:
            name (str): The name of the computation (must have a matching method name inside `component`).
            component (Component): The Component object that this Computation belongs to.
            input_sockets (list): The required input Sockets to be passed as parameters into the computation function.
            output_sockets (list): The Sockets associated with the return values coming from the computation function.
        """
        # The raw computation-method (not autograph'd yet).
        self.name = name
        self.component = component

        # Derive precomputation and computation methods from name and build the computation method.
        # TODO: Discuss namings of the following two items:
        self.pre_method = self.component.__getattribute__("_pre_" + self.name)
        self.method = self.component.__getattribute__("_computation_" + self.name)
        self.graph_method = self.to_graph(method=self.method)

        self.input_sockets = input_sockets
        self.output_sockets = output_sockets

        # Registry for which incoming Sockets we already "have waiting".
        self.waiting_ops = dict()  # key = Socket object; value = list of ops the key (Socket) carries.

        # Whether we have all necessary input-sockets for passing at least one input-op combination through
        # our computation method. As long as this is False, we return prematurely and wait for more ops to come in
        # (through other Sockets).
        self.input_complete = False
        # Registry for which incoming Sockets' ops we have already passed through the computation to generate
        # output ops.
        # key=tuple of input-ops combination (len==number of input params).
        # value=list of generated output ops (len==number of return values).
        self.processed_ops = dict()

    def to_graph(self, method):
        """
        Converts function containing Python control flow to graph.

        Args:
            method (callable): Function object containing computations and potentially control flow.

        Returns:
            Computation graph object.
        """
        return method  # not mandatory

    def update_from_input(self, input_socket, op_registry, socket_registry):
        """
        Updates our "waiting" inputs with the incoming socket and checks whether this computation is "input-complete".
        If yes, do all possible combinatorial pass-throughs through the computation function to generate output ops
        and assign these ops to our respective output sockets (first socket gets first output op, etc.. depending
        on how many return values the computation function has).

        Args:
            input_socket (Socket): The incoming Socket (by design, must be type "in").
            op_registry (dict): Dict that keeps track of which ops require which other ops to be calculated.
            socket_registry (dict): Dict that keeps track of which in-Socket needs which ops (placeholders/feeds).
        """
        assert isinstance(input_socket, Socket) and input_socket.type == "in", \
            "ERROR: `input_socket` must be a Socket object and of type 'in'!"
        # Update waiting_ops.
        if input_socket.name not in self.waiting_ops:
            self.waiting_ops[input_socket.name] = set()
        self.waiting_ops[input_socket.name].update(input_socket.ops)
        # If "input-complete", pass through computation function.
        if not self.input_complete:
            # Check, whether we are input-complete now (whether all required inputs are given).
            self.input_complete = True
            for required in self.input_sockets:
                if required.name not in self.waiting_ops:
                    self.input_complete = False
                    break

        # No elif! We have to check again.
        # We are input-complete: Get all possible combinations of input ops and pass all these combinations through
        # the function (only those combinations that we didn't do yet).
        if self.input_complete:
            # A list of all possible input op combinations.
            input_combinations = list(itertools.product(*self.waiting_ops.values()))
            for input_combination in input_combinations:
                # key = tuple(input_combination)
                # Make sure we call the computation method only once per input-op combination.
                if input_combination not in self.processed_ops:
                    # Call our pre-computation method first.
                    returns = self.pre_method(*input_combination)
                    ops = self.graph_method(self.component, *returns)
                    self.processed_ops[input_combination] = tuple(ops)
                    # Keep track of which ops require which other ops.
                    op_registry[tuple(ops)] = input_combination

            # Loop through our output Sockets and keep processing them with this computation's outputs.
            for slot, output_socket in enumerate(self.output_sockets):
                output_socket.update_from_input(self, op_registry, socket_registry, slot)

    def __str__(self):
        return "{}('{}' in=[{}] out=[{}])". \
            format(type(self).__name__, self.method.__name__, str(self.input_sockets), str(self.output_sockets))


class TfComputation(Computation):
    """
    TensorFlow computation.
    """
    def to_graph(self, method):
        return autograph.to_graph(method, verbose=True)

