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
import itertools

from yarl import YARLError, Specifiable
from yarl.components import Component, Socket, Computation
from yarl.utils.util import all_combinations, force_list, get_shape
from yarl.utils.input_parsing import parse_saver_spec, parse_summary_spec, parse_execution_spec


class Model(Specifiable):
    """
    Contains:
    - actual tf.Graph
    - tf.Session (which is wrapped by a simple, internal API that the Algo can use to fetch ops from the graph).
    - Savers, Summarizers
    - The core Component object (with all its sub-Components).

    The Agent/Algo can use the Model's core-Component's exposed Sockets to reinforcement learn.
    """
    def __init__(self, name="model", saver_spec=None, summary_spec=None, execution_spec=None):
        """
        Args:
            name (str): The name of this model.
            saver_spec (dict): The saver specification for saving this graph to disk.
            summary_spec (dict): The specification dict for summary generation.
            execution_spec (dict): The specification dict for the execution types (local vs distributed, etc..) and
                settings (cluster types, etc..).
        """
        # The name of this model. Our core Component gets this name.
        self.name = name
        self.saver_spec = parse_saver_spec(saver_spec)
        self.summary_spec = parse_summary_spec(summary_spec)
        self.execution_spec = parse_execution_spec(execution_spec)  # sanitize again (after Agent); one never knows
        # Default single-process execution.
        self.execution_mode = self.execution_spec.get("mode", "single")
        self.seed = self.execution_spec.get("seed")

        self.session_config = self.execution_spec["session_config"]
        self.distributed_spec = self.execution_spec.get("distributed_spec")

        # Create an empty core Component into which everything will be assembled by an Algo.
        self.core_component = Component(name=self.name)
        # List of variables (by scope/name) of all our components.
        self.variables = dict()
        # A dict used for lookup of all combinations that are possible for a given set of given in-Socket
        # names (inside a call to `self.call`).
        self.input_combinations = dict()
        # Some registries that we need to build the Graph from core.
        # key=op; value=list of required ops to calculate the key-op
        self.op_registry = dict()
        # key=in-Socket name; value=set of alternative ops (dictop/tuple/op) that could go into this socket.
        # Only for very first in-Sockets.
        self.in_socket_registry = dict()
        # Maps an out-Socket name+in-Socket/Space-combination to an actual op to fetch from our Graph.
        self.call_registry = dict()  # key=()

        # Computation graph.
        self.graph = None

    def build(self):
        """
        Sets up the computation graph by:
        - Starting the Server, if necessary.
        - Setting up the computation graph object.
        - Assembling the computation graph defined inside our core component.
        - Setting up graph-savers, -summaries, and finalizing the graph.
        """
        # Starts the Server (if in distributed mode).
        # If we are a ps -> we stop here and just run the server.
        self.init_execution()

        # Creates the Graph.
        self.setup_graph()
        # Loops through all our components and assembles the graph.
        self.assemble_graph()

        # Set up any remaining session or monitoring configurations.
        self.finalize_backend()

    def call(self, sockets, inputs=None):
        """
        Fetches one or more Socket outputs from the graph (given some inputs) and returns their outputs.

        Args:
            sockets (Union[str,List[str]]): A name or a list of names of the (out) Sockets to fetch from our core
                component.
            inputs (Optional[dict]): Dict specifying the provided inputs for some (in) Sockets. Depending on these
                given inputs, the correct backend-ops can be selected within the given (out) sockets.

        Returns:
            Tuple (or single item) containing the results from fetching all the given out-Sockets.
        """
        raise NotImplementedError

    def init_execution(self):
        """
        Sets up backend-dependent execution, e.g. server for distributed TensorFlow
        execution.
        """
        pass  # not mandatory

    def setup_graph(self):
        """
        Creates the computation graph.
        """
        # TODO mandatory?
        raise NotImplementedError

    def assemble_graph(self):
        """
        Loops through all our sub-components starting at core and assembles the graph by creating placeholders,
        following Socket->Socket connections and running through our Computations.
        """
        # Loop through the given input sockets and connect them from left to right.
        for socket in self.core_component.input_sockets:
            # sanity check our input Sockets for connected Spaces.
            if len(socket.incoming_connections) == 0:
                raise YARLError("ERROR: Exposed Socket '{}' does not have any connected (incoming) Spaces!".
                                format(socket))
            self.partial_input_build(socket)

        # Memoize possible input-combinations (from all our in-Sockets)
        # so we don't have to do this every time we get a `call`.
        in_names = sorted(list(map(lambda s: s.name, self.core_component.input_sockets)))
        input_combinations = all_combinations(in_names, descending_length=True)
        # Store each combination and its sub-combinations in self.input_combinations.
        for input_combination in input_combinations:
            self.input_combinations[tuple(input_combination)] = \
                all_combinations(input_combination, descending_length=True)

        # Now use the ready op/socket registries to determine for which out-Socket we need which inputs.
        # Then we will be able to derive the correct op for any given (out-Socket+in-Socket+in-shape)-combination
        # passed into the call method.
        for output_socket in self.core_component.output_sockets:
            # Loop through this Socket's set of possible ops.
            for op in output_socket.ops:
                # Get all the (core) in-Socket names (alphabetically sorted) that are required for this op.
                sockets = tuple(sorted(list(self.trace_back_sockets({op})), key=lambda s: s.name))
                # If an in-Socket has more than one connected incoming Space:
                # Get the shape-combinations for these Sockets.
                # e.g. Sockets=["a", "b"] (and Space1 -> a, Space2 -> a, Space3 -> b)
                #   shape-combinations=[(Space1, Space3), (Space2, Space3)]
                shapes = [[i.shape_with_batch_rank for i in sock.incoming_connections] for sock in sockets]
                shape_combinations = itertools.product(*shapes)
                for shape_combination in shape_combinations:
                    # Do everything by Socket-name (easier to debug).
                    key = (output_socket.name, tuple([s.name for s in sockets]), shape_combination)
                    self.call_registry[key] = op

    def finalize_backend(self):
        """
        Initializes any remaining backend-specific monitoring or session handling.
        """
        raise NotImplementedError

    def load_model(self, path=None):
        """
        Loads model from specified path location.

        Args:
            path (str): Path to checkpoint or model.
        """
        raise NotImplementedError

    def store_model(self, path=None, add_timestep=True):
        """
        Saves the model to the given path (or to self.saver_directory). Optionally adds the current timestep
        to the filename to prevent overwriting previous checkpoint files.

        Args:
            path (str): The directory in which to save (default: self.saver_directory).
            add_timestep: Appends the current timestep to the checkpoint file if true.
        """
        raise NotImplementedError

    def get_default_model(self):
        """
        Returns:
            Component: The core container component.
        """
        return self.core_component

    def partial_input_build(self, socket, from_=None, computation_out_slot=None, socket_out_op=None):
        """
        Builds one Socket in terms of its incoming and outgoing connections. Returns if we hit a dead end (computation
        that's not complete yet) OR the very end of the graph.
        Recursively calls itself for any encountered Sockets on the way through the Component(s).

        Args:
            socket (Socket): The Socket object to process.
            from_ (Optional[Computation,Socket]): If we are processing the socket only from one Computation/Socket
                (ignoring other incoming connections), set this to the Computation/Socket to build from.
            computation_out_slot (Optional[int]): If from_ is a Computation, this holds the out slot from the
                Computation (index into the computation-returned tuple).
            socket_out_op (Optional[op]): If from_ is a Socket, this holds the op from the from_-Socket that should be
                built from.
        """
        with tf.variable_scope(socket.component.scope):
            # Loop through this socket's incoming connections (or just process from_).
            incoming_connections = from_ or socket.incoming_connections
            for in_ in force_list(incoming_connections):
                # create tf placeholder(s)
                # example: Space (Dict({"a": Discrete(3), "b": Bool(), "c": Continuous()})) connects to Socket ->
                # create inputs[name-of-sock] = dict({"a": tf.placeholder(name="", dtype=int, shape=(3,))})
                socket.update_from_input(in_, self.op_registry, self.in_socket_registry,
                                         computation_out_slot, socket_out_op)

            for outgoing in socket.outgoing_connections:
                # Outgoing is another Socket (other Component) -> recurse.
                if isinstance(outgoing, Socket):
                    # This component is already complete. Do only a partial build.
                    if outgoing.component.input_complete:
                        self.partial_input_build(outgoing, from_=socket, socket_out_op=socket_out_op)
                    # Component is not complete yet:
                    else:
                        # Add one Socket (Space) information.
                        outgoing.update_from_input(socket, self.op_registry, self.in_socket_registry)
                        # Check now for input-completeness of this component.
                        if outgoing.component.input_complete:
                            # Create the Component's variables.
                            space_dict = {in_s.name: in_s.space for in_s in outgoing.component.input_sockets}
                            outgoing.component.create_variables(space_dict)
                            # Do a complete build (over all incoming Sockets as some of these have been waiting).
                            self.partial_input_build(outgoing)

                # Outgoing is a Computation -> Add the socket to the computations (waiting) inputs.
                # - If all inputs are complete, build a new op into the graph (via the computation).
                elif isinstance(outgoing, Computation):
                    computation = outgoing
                    # We have to specify the device here (only a Computation actually adds something to the graph).
                    if socket.component.device:
                        self.assign_device(computation, socket, socket.component.device)
                    else:
                        # TODO fetch default device?
                        computation.update_from_input(socket, self.op_registry, self.in_socket_registry)

                    # Keep moving through this computations out-Sockets (if input-complete).
                    if computation.input_complete:
                        for slot, out_socket in enumerate(computation.output_sockets):
                            self.partial_input_build(out_socket, computation, slot)
                else:
                    raise YARLError("ERROR: Outgoing connection must be Socket or Computation!")

    def get_execution_inputs(self, socket_names, input_dict=None):
        """
        Fetches graph inputs for execution.

        Args:
            socket_names (Union[str,List[str]]): A name or a list of names of the (out) Sockets to fetch from our core
                component.
            input_dict (Optional[dict]): Dict specifying the provided inputs for some (in) Sockets.
                Depending on these given inputs, the correct backend-ops can be selected within the given (out)-Sockets.

        Returns:
            Input-list and feed-dict with relevant args.
9       """

        # Try all possible input combinations to see whether we got an op for that.
        # Input Socket names will be sorted alphabetically and combined from short sequences up to longer ones.
        # Example: input_dict={A: ..., B: ... C: ...}
        #   input_combinations=[ABC, AB, AC, BC, A, B, C]

        # These combinations have been memoized for fast lookup.
        key = tuple(sorted(input_dict.keys()))
        input_combinations = self.input_combinations.get(key)
        if not input_combinations:
            raise YARLError("ERROR: Could not find input_combinations for in-Sockets '{}'!".format(key))

        # Go through each (core) out-Socket names and collect the correct ops to go into the fetch_list.
        fetch_list = list()
        feed_dict = dict()
        for socket_name in socket_names:
            self._get_execution_inputs_for_socket(socket_name, input_combinations, fetch_list, input_dict, feed_dict)

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
            input_dict (Optional[dict]): Dict specifying the provided inputs for some (core) in-Sockets.
                Passed through directly from the call method.
            feed_dict (dict): The feed_dict we are trying to build. When done, needs to map input ops (not Socket names)
                to data.
        """
        # Check all (input+shape)-combinations and it we find one that matches what the user passed in as
        # `input_dict` -> Take that one and move on to the next Socket by returning.
        for input_combination in input_combinations:
            # Get all Space-combinations (in-op) for this input combination
            # (in case an in-Socket has more than one connected incoming Spaces).
            ops = [self.in_socket_registry[c] for c in input_combination]
            op_combinations = itertools.product(*ops)
            for op_combination in op_combinations:
                # Get the shapes for this op_combination.
                shapes = tuple(get_shape(op) for op in op_combination)
                key = (socket_name, input_combination, shapes)
                # This is a good combination -> Use the looked up op, return to process next out-Socket.
                if key in self.call_registry:
                    fetch_list.append(self.call_registry[key])
                    # Store for which in-Socket we need which in-op to put into the feed_dict.
                    for in_sock_name, in_op in zip(input_combination, op_combination):
                        # TODO: in_op may still be a dict or a tuple depending on what Space underlies.
                        # Need to split into single ops.
                        feed_dict[in_op] = input_dict[in_sock_name]
                    return

        raise YARLError("ERROR: No op found for out-Socket '{}' given the input-combinations: {}!".
                        format(socket_name, input_combinations))

    def trace_back_sockets(self, trace_set):
        """
        For a set of given ops, returns a list of all (core) in-Sockets that are required to calculate these ops.

        Args:
            trace_set (set): The set of ops to trace-back till the beginning of the Graph.

        Returns:
            set: in-Socket objects (from the core Component) that are required to calculate this op.
        """
        # Recursively lookup op in op_registry until we hit a Socket.
        new_trace_set = set()
        for op in trace_set:
            if op not in self.op_registry:
                if op not in self.in_socket_registry:
                    raise YARLError("ERROR: op {} could not be found in op_registry or in_socket_registry of model!".
                                    format(op.name))
                # Already a Socket -> add to new set and continue.
                else:
                    new_trace_set.add(op)
                    continue
            new_trace_set.update(self.op_registry[op])
        if all([isinstance(i, Socket) for i in new_trace_set]):
            return new_trace_set
        else:
            return self.trace_back_sockets(new_trace_set)

    def assign_device(self, computation, socket, assigned_device):
        """
        Assigns device to socket.
        Args:
            computation (Computation): Computation to assign device to.
            socket (Socket): Socket used on Computation
            assigned_device (str): Device identifier.
        """
        # If this is called, the backend should implement it, otherwise device
        # would be ignored -> no pass here.
        raise NotImplementedError

