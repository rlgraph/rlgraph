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
#from tensorflow.contrib import autograph
from collections import OrderedDict
import re

from yarl import YARLError
from yarl.spaces import Space, get_space_from_op
from yarl.utils.dictop import dictop
from yarl.utils.util import force_tuple, get_shape, deep_tuple


class Socket(object):
    """
    A Socket object describes a connection to other Sockets, Computations, or Spaces inside and between ModelComponents.
    One Socket either carries:

    - a single op (e.g. some tensor)
    - a tuple of ops (nesting also supported)
    - a dict of ops (nesting also supported)

    Also, each of the above possibilities can have many parallel solutions. These splits happen e.g. if two Sockets
    connect to the same target Socket. In this case, the target Socket's inputs are treated as possible alternatives
    and the Socket then implicitly produces two outputs that it further passes on to the next Sockets/Computations.

    When connected to a computation object, a Socket always represents one of the input parameters to the computation
    method. Also, each returned value of a computation method corresponds to one Socket.
    """
    def __init__(self, name, component, type_="in"):
        """
        Args:
            name (str): The name of this Socket (as it will show in the final call interface).
            component (Component): The Component object that this Socket belongs to.
            type_ (str): The Socket type: "in" or "out".
            #OBSOLETE: we have the component now!
            #scope (str): The scope of this Socket (same as the owning Component's scope).
            #device (str): Device this Socket's component will be assigned to. If None, defaults to CPU.
            #global_ (bool): In distributed mode, this flag indicates if the Socket's component is part of the
            #    shared global model or local to the worker. Defaults to False and will be ignored if set to
            #    True in non-distributed mode.
        """
        # The name of this Socket.
        self.name = name
        # "in" or "out"
        self.type = type_
        # The Component that this Socket belongs to.
        self.component = component

        ## TODO: remove the following again and get them from the component. Store the component in the Socket.
        ## The scope of this socket (important for computations and variable creation).
        #self.scope = scope
        ## The device to use when placing ops that come after this Socket into the Graph.
        #self.device = device
        #self.global_ = global_

        # Which other socket(s), space(s), computation(s) are we connected to on the incoming and outgoing side?
        # - Records in these lists have a parallel relationship (they are all alternatives to each other).
        # - Each record is either:
        #   - another Socket
        #   - a Space
        #   - a dict of: {computation-method, slot for this socket, [required input sockets], [output sockets]}
        self.incoming_connections = list()
        self.outgoing_connections = list()

        # The inferred Space coming into this Socket.
        # TODO: Make sure all incoming connections have the same Space.
        self.space = None

        # The set of (alternative) ops (dictop, tuple or primitive) that this socket carries. Populated at build time.
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

    def update_from_input(self, incoming, op_registry, in_socket_registry, computation_in_slot=None,
                          socket_in_op=None):
        """
        Updates this socket based on an incoming connection (from a Space or Computation or another Socket).

        Args:
            incoming (Union[Space,Computation,Socket]): The incoming item.
            op_registry (dict): Dict that keeps track of which ops require which other ops to be calculated.
            in_socket_registry (dict): Dict that keeps track of which very in-Socket (name) needs which
                ops (placeholders/feeds).
            computation_in_slot (Optional[int]): If incoming is a Computation, which output slot does this Socket
                connect to?
            socket_in_op (Optional[op]): If incoming is a Socket, this may hold a single op that we should
                build from. If None and incoming is Socket, update all.

        Raises:
            YARLError: If there is an attempt to connect more than one Space to this Socket.
        """
        # Space: generate backend-ops.
        if isinstance(incoming, Space):
            if self.space is not None:
                raise YARLError("ERROR: A Socket can only have one incoming Space!")
            # TODO: Add batch dimension-option.
            # NOTE: op could be dictop or tuple as well.
            op = incoming.get_tensor_variable(name=self.name, is_input_feed=True)
            # Add new op to our set of (alternative) ops.
            self.ops.add(op)
            # Keep track of which Spaces can go (alternatively) into this Socket.
            in_socket_registry[self.name] = {op} if self.name not in in_socket_registry \
                else (in_socket_registry[self.name] | {op})
            # Remember, that this op goes into a Socket at the very beginning of the Graph (e.g. a tf.placeholder).
            op_registry[op] = {self}
            # Store this Space as our incoming Space.
            self.space = incoming
        # Computation: Connect this Socket to the nth op coming out of the Computation function.
        elif isinstance(incoming, Computation):
            assert isinstance(computation_in_slot, int) and computation_in_slot >= 0, \
                "ERROR: If incoming is a Computation, slot must be set and >=0!"
            # Add every nth op from the output of the completed computations to this Socket's set of ops.
            nth_computed_ops = [outputs[computation_in_slot] for outputs in incoming.processed_ops.values()]

            # Store incoming Space.
            if len(self.ops) == 0:
                in_space = get_space_from_op(next(iter(nth_computed_ops)))
                # TODO: check whether all incoming ops have same Space
                self.space = in_space

            self.ops.update(nth_computed_ops)
        # Incoming is another Socket -> Simply update ops from this one.
        else:
            assert isinstance(incoming, Socket), "ERROR: Incoming must be Space, Computation, or another Socket!"
            # Update given op or all (it's a set, so no harm).
            self.ops.update(socket_in_op or incoming.ops)
            self.space = incoming.space
            # Check whether our Component now has all it's Sockets with a Space-information.
            self.component.input_complete = True
            for in_sock in self.component.input_sockets:
                if in_sock.space is None:
                    self.component.input_complete = False
                    break

    def __str__(self):
        return "{}-Socket('{}/{}'{})".format(self.type, self.component.scope, self.name,
                                             " dev='{}'".format(self.component.device) if self.component.device else "")


class Computation(object):
    """
    Class describing a computation happening inside a _computation-func inside a Component.
    Implements the update_from_input method which checks whether all necessary inputs to a computation function
    are given and - if yes - starts producing output ops from these inputs and the computation to be passed
    on to the outgoing Sockets.
    """
    def __init__(self, name, component, input_sockets, output_sockets,
                 flatten_container_spaces=True, split_container_spaces=True,
                 add_auto_key_as_first_param=False, re_nest_container_spaces=True):
        """
        Args:
            name (str): The name of the computation (must have a matching method name inside `component`).
            component (Component): The Component object that this Computation belongs to.
            input_sockets (List[Socket]): The required input Sockets to be passed as parameters into the
                computation function.
            output_sockets (List[socket]): The Sockets associated with the return values coming from the computation
                function.
            flatten_container_spaces (Union[bool,Set[str]]): Whether to flatten input ContainerSpaces by creating
                a flat OrderedDict (with automatic key names) out of Dict/Tuple.
                Alternatively, can be a set of in-Socket names to flatten.
                (default: True).
            split_container_spaces (Union[bool,Set[str]]): Whether to flatten, then split all or some input
                ContainerSpaces and send the single, primitive Spaces one by one through the computation.
                In this case, the computation function needs to expect single, primitive Spaces as input parameters.
                Example: in-Sockets=A=Dict (container), B=int (primitive)
                    The computation func should then expect for each primitive Space in A:
                        _computation_func(primitive-in-A (Space), B (int))
                        NOTE that B will be the same in all calls for all primitive-in-A's.
                (default: True).
            add_auto_key_as_first_param (bool): If `split_container_spaces` is not False, whether to send the
                automatically generated flat key as the very first parameter into each call of the computation func.
                Example: in-Sockets=A=float (primitive), B=Tuple (container)
                    The computation func should then expect for each primitive Space in B:
                        _computation_func(key, A (float), primitive-in-B (Space))
                        NOTE that A will be the same in all calls for all primitive-in-B's.
                        The key can now be used to index into variables equally structured as B.
                Has no effect if `split_container_spaces` is False.
                (default: False).
            re_nest_container_spaces (bool): Whether to re-establish the originally nested structure of computations
                for an incoming, flattened ContainerSpace.
                Only relevant if flatten_container_spaces is not False.
                (default: True)

        Raises:
            YARLError: If a computation method with the given name cannot be found in the component.
        """

        # Must match the _computation_...-method's name (w/o the `_computation`-prefix).
        self.name = name
        # The component object that the method belongs to.
        self.component = component

        self.flatten_container_spaces = flatten_container_spaces
        self.split_container_spaces = split_container_spaces
        self.add_auto_key_as_first_param = add_auto_key_as_first_param
        self.re_nest_container_spaces = re_nest_container_spaces

        self.method = getattr(self.component, "_computation_" + self.name, None)
        if not self.method:
            raise YARLError("ERROR: No `_computation_...` method with name '{}' found!".format(self.name))

        #self.create_variables = getattr(self.component, "create_variables", None)
        #if not self.create_variables:
        #    raise YARLError("ERROR: No `create_variables` method found in component '{}'!".format(self.component.name))

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

    #def to_graph(self, method):
    #    """
    #    Converts function containing Python control flow to graph.
    #
    #    Args:
    #        method (callable): Function object containing computations and potentially control flow.
    #
    #    Returns:
    #        Computation graph object.
    #    """
    #    return method  # not mandatory

    def update_from_input(self, input_socket, op_registry, in_socket_registry):
        """
        Updates our "waiting" inputs with the incoming socket and checks whether this computation is "input-complete".
        If yes, do all possible combinatorial pass-throughs through the computation function to generate output ops
        and assign these ops to our respective output sockets (first socket gets first output op, etc.. depending
        on how many return values the computation function has).

        Args:
            input_socket (Socket): The incoming Socket (by design, must be type "in").
            op_registry (dict): Dict that keeps track of which ops require which other ops to be calculated.
            in_socket_registry (dict): Dict that keeps track of which in-Socket (name) needs which
                ops (placeholders/feeds).
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
                    # Build the ops from this input-combination.
                    # - Flatten complex spaces.
                    if self.flatten_container_spaces is not False:
                        flattened_spaces = self.flatten_container_ops(*input_combination)
                        if self.split_container_spaces:
                            # ops is an OrderedDict of return-tuples.
                            ops = self.split_flattened_ops(*flattened_spaces)
                        else:
                            # ops is a return-tuple (maybe including flattened OrderedDicts) or a single op.
                            ops = self.method(*input_combination)

                        # Need to re-nest?
                        if self.re_nest_container_spaces:
                            ops = self.re_nest_container_ops(*force_tuple(ops))

                    # - Simple case: Just pass in everything as is.
                    else:
                        ops = self.method(*input_combination)

                    # Make sure everything coming from a computation is always a tuple (for out-Socket indexing).
                    ops = force_tuple(ops)

                    self.processed_ops[input_combination] = ops
                    # Keep track of which ops require which other ops.
                    for op in ops:
                        op_registry[op] = set(input_combination)

            # Loop through our output Sockets and keep processing them with this computation's outputs.
            for slot, output_socket in enumerate(self.output_sockets):
                output_socket.update_from_input(self, op_registry, in_socket_registry, slot)

    def flatten_container_ops(self, *ops):
        """
        Flattens all ContainerSpace instances in ops into python OrderedDicts with auto-key generation.
        Primitives ops and ops whose Sockets are not in self.flatten_container_spaces (if its a set)
        will be ignored.

        Args:
            *ops (Union[dictop,tuple,op]): The ops to flatten.

        Returns:
            tuple: All *ops as flattened OrderedDicts (or left unchanged if already primitive).
        """
        # The returned sequence of output ops.
        ret = []

        for i, op in enumerate(ops):
            # The in-Socket name of this op.
            socket_name = self.input_sockets[i].name
            # self.flatten_container_spaces cannot be False here.
            if isinstance(op, (dictop, tuple)) and \
                    (self.flatten_container_spaces is True or socket_name in self.flatten_container_spaces):
                ret.append(self._flatten_container_op(op))
            # Primitive ops are left as-is.
            else:
                ret.append(op)

        # Always return a tuple for indexing into the return values.
        return tuple(ret)

    def _flatten_container_op(self, op, scope_="", list_=None):
        """
        Flattens a single ContainerSpace (op) into a python OrderedDict with auto-key generation.

        Args:
            op (Union[dictop,tuple]): The op to flatten. This can only be a tuple or a dictop.
            scope_ (str): The recursive scope for auto-key generation.
            list_ (list): The list of tuples (key, value) to be converted into the final OrderedDict.

        Returns:
            OrderedDict: The flattened representation of the op.
        """
        ret = False
        # Are we in the non-recursive (first) call?
        if list_ is None:
            assert isinstance(op, (dictop, tuple)), "ERROR: Can only flatten container (dictop/tuple) ops!"
            list_ = list()
            ret = True

        if isinstance(op, tuple):
            scope_ += "/["
            for i, c in enumerate(op):
                self._flatten_container_op(c, scope_=scope_ + str(i) + "]", list_=list_)
        elif isinstance(op, dictop):
            scope_ += "/"
            for k, v in op.items():
                self._flatten_container_op(v, scope_=scope_ + k, list_=list_)
        else:
            list_.append((scope_, op))

        # Non recursive (first) call -> Return the final OrderedDict.
        if ret:
            return OrderedDict(list_)

    def split_flattened_ops(self, *ops):
        """
        Splits any (flattened) OrderedDict-type op in ops into its single, primitive ops and passes them
        one by one through the computation function. If more than one container op exists in ops,
        these must have the exact same key/value structure and sequence.
        If self.add_auto_key_as_first_param is True: Pass in auto-key as very first parameter into each
            call to computation func.

        Args:
            *ops (any): The input ops into this Computation. Some of these may be flattened container ops (OrderedDict).

        Returns:
            OrderedDict: The flattened, sorted, generated ops (if at least one in ops is already a flattened
                OrderedDict.
            tuple(ops): If no flattened OrderedDict is in ops.

        Raises:
            YARLError: If there are more than 1 flattened ops in ops and their keys and value-types don't match 100%.
        """
        # Collect OrderedDicts (these are the flattened ones) for checking their structures (must match).
        flattened = [op.items() for op in ops if isinstance(op, OrderedDict)]
        # If it's more than 1, make sure they match. If they don't match: raise Error.
        if len(flattened) > 1:
            # Loop through the first one and make sure all others match.
            for key, value in flattened[0]:
                for other in flattened[1:]:
                    k_other, v_other = next(other)
                    if k_other != key or get_shape(v_other) != get_shape(value):
                        raise YARLError("ERROR: Flattened ops dont match in structure (key={})!".format(key))

        # We have (matching) container ops: Split the calls.
        if len(flattened) > 0:
            # The first op that is an OrderedDict.
            guide_op = next(op for op in ops if isinstance(op, OrderedDict))
            # Re-create our iterators.
            flattened = [op.items() if isinstance(op, OrderedDict) else op for op in ops]
            collected_returns = OrderedDict()
            # Do the single split calls to our computation func.
            for key, value in guide_op.items():
                # Prep input params for a single call.
                params = [key, value] if self.add_auto_key_as_first_param else [value]
                # Pull along the other ops' values for the guide_op's current key
                # (all container ops match structure-wise).
                for other in flattened[1:]:
                    v_other = next(other)[1] if isinstance(other, OrderedDict) else other
                    params.append(v_other)
                # Now do the single call.
                collected_returns[key] = self.method(*params)

            return collected_returns
        # We don't have any container ops: No splitting possible. Return as is.
        else:
            return force_tuple(self.method(*ops))

    def re_nest_container_ops(self, *ops, input_comparison_op=None):
        """
        Re-creates the originally nested input structure (as dict/tuple) of the given output ops.
        Process all flattened OrderedDicts with auto-generated keys, and leave the others (primitives)
        untouched.

        Args:
            *ops (Union[OrderedDict,dictop,tuple,op]): The ops that need to be re-nested (only process the OrderedDicts
                amongst these and ignore all others).
            TODO: input_comparison _op (Optional[OrderedDict]): One of the flattened input ops to use for sanity checking.
                If not None, all output OrderedDict ops must match this one's key/value structure.

        Returns:
            Tuple[Union[dict,tuple,op]]: A tuple containing the ops as they came in, except that all OrderedDicts
                have been re-nested into their original (dict/tuple) container structures.
        """
        # The returned sequence of output ops.
        ret = []

        for i, op in enumerate(ops):
            # An OrderedDict: Try to re-nest it and then compare it to input_template_op's structure.
            if isinstance(op, OrderedDict):
                ret.append(self._re_nest_container_op(op))  #, input_comparison_op=input_comparison_op))
            # All others are left as-is.
            else:
                ret.append(op)

        # Always return a tuple for indexing into the return values.
        return tuple(ret)

    @staticmethod
    def _re_nest_container_op(op):
        base_structure = None

        for k, v in op.items():
            parent_structure = None
            parent_key = None
            current_structure = None
            type_ = None

            keys = k[1:].split("/")  # skip 1st char (/)
            for key in keys:
                mo = re.match(r'^\[(\d+)\]$', key)
                if mo:
                    type_ = list
                    idx = int(mo.group(1))
                else:
                    type_ = dictop
                    idx = key

                if current_structure is None:
                    if base_structure is None:
                        base_structure = [None] if type_ == list else dictop()
                    current_structure = base_structure
                elif parent_key is not None:
                    if isinstance(parent_structure, list) and parent_structure[parent_key] is None or \
                            isinstance(parent_structure, dictop) and parent_key not in parent_structure:
                        current_structure = [None] if type_ == list else dictop()
                        parent_structure[parent_key] = current_structure
                    else:
                        current_structure = parent_structure[parent_key]
                        if type_ == list and len(current_structure) == idx:
                            current_structure.append(None)

                parent_structure = current_structure
                parent_key = idx

            if type_ == list and len(current_structure) == parent_key:
                current_structure.append(None)
            current_structure[parent_key] = v

        # Deep conversion from list to tuple.
        return deep_tuple(base_structure)

    def __str__(self):
        return "{}('{}' in=[{}] out=[{}])". \
            format(type(self).__name__, self.name, str(self.input_sockets), str(self.output_sockets))


#class TfComputation(Computation):
#    """
#    TensorFlow computation.
#    """
#    def to_graph(self, method):
#        return autograph.to_graph(method, verbose=True)

