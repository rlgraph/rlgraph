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

from yarl import YARLError, backend
from yarl.components import Component, Socket, Computation
from yarl.utils.util import all_combinations
from yarl.utils.input_parsing import parse_execution_spec


class Model(object):
    """
    Contains:
    - actual tf.Graph
    - tf.Session (which is wrapped by a simple, internal API that the Algo can use to fetch ops from the graph).
    - Savers, Summarizers
    - The core Component object (with all its sub-ModelComponents).

    The Agent/Algo can use the Model's core-Component's exposed Sockets to reinforcement learn.
    """
    def __init__(self, name="model", saver_spec=None, summary_spec=None, execution_spec=None):
        """
        Args:
            name (str): The name of this model.
            summary_spec (dict): The specification dict for summary generation.
            saver_spec (dict): The saver specification for saving this graph to disk.
        """
        # The name of this model. Our core Component gets this name.
        self.name = name
        self.saver_spec = saver_spec
        self.summary_spec = summary_spec
        self.execution_spec = parse_execution_spec(execution_spec)  # sanitize again (after Agent); one never knows
        # Default single-process execution.
        self.execution_mode = self.execution_spec.get("mode", "single")
        self.session_config = self.execution_spec["session_config"]
        self.distributed_spec = self.execution_spec.get("distributed_spec")

        # Create an empty core Component into which everything will be assembled by an Algo.
        self.core_component = Component(name=name)
        # List of variables (by scope/name) of all our components.
        self.variables = {}
        # Some registries that we need to build the Graph from core.
        # key=op; value=list of required ops to calculate the key-op
        self.op_registry = {}
        # key=Socket; value=list of alternative placeholders that could go into this socket.
        # Only for very first (in) Sockets.
        self.socket_registry = {}
        # Maps a out-Socket name+in-Socket/Space-combination to an actual op to fetch from our Graph.
        self.call_registry = {}  # key=()

        # Computation graph.
        self.graph = None

    def build(self):
        """
        Sets up the computation graph by:
        - Starting the Server, if necessary.
        - Setting up the computation graph object.
        - Assembling the computation graph defined inside our core component.
        - Setting up Savers and Summaries.
        -
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

    def call(self, sockets, input=None):
        """
        Fetches one or more Socket outputs from the graph (given some inputs) and returns their outputs.

        Args:
            sockets (Union[str,List[str]]): A name or a list of names of the (out) Sockets to fetch from our core
                component.
            input (Union[dict,None]): Dict specifying the provided inputs for some (in) Sockets. Depending on these
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
        following Socket->Socket connections and running the autograph functions of our Computations.
        """
        # Loop through our input sockets and connect them from left to right.
        for socket in self.core_component.input_sockets:
            # sanity our input Sockets ofr connected Spaces.
            if len(socket.incoming_connections) == 0:
                raise YARLError("ERROR: Exposed Socket '{}' does not have any connected (incoming) Spaces!".
                                format(socket))
            self.partial_input_build(socket)

        # Now use the ready op/socket registries to determine for which out-Socket we need which inputs.
        # Then we will be able to derive the correct ops for any given out-Socket+input-Sockets combination
        # in the call method.
        for output_socket in self.core_component.output_sockets:
            # Loop through this Socket's set of possible ops.
            for op in output_socket.ops:
                # Get all the Sockets that lead to this op.
                sockets = self.trace_back_sockets({op})
                all_combinations = itertools.product(*sockets)
                # store all possible combinations in call_registry (for convenience).
                for combination in all_combinations:
                    key = tuple([output_socket.name, tuple(all_combinations)])
                    self.call_registry[key] = combination

    def finalize_backend(self):
        """
        Initializes any remaining backend-specific monitoring or session handling.
        """
        raise NotImplementedError()

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
        Returns the default container.

        Returns: The model's default component.
        """
        return self.core_component

    def get_execution_inputs(self, sockets, input_dict=None):
        """
        Fetches graph inputs for execution.

        Args:
            sockets (Union[str,List[str]]): A name or a list of names of the (out) Sockets to fetch from our core
                component.
            input_dict (Union[dict,None]): Dict specifying the provided inputs for some (in) Sockets.
                Depending on these given inputs, the correct backend-ops can be selected within the given (out)-Sockets.

        Returns:
            Input-list and feed-dict with relevant args.
        """
        fetch_list = []
        feed_dict = {}

        # Try all possible input combinations to see whether we got an op for that.
        input_combinations = all_combinations(input_dict, descending=True)

        for socket in sockets:
            for input_combination in input_combinations:
                key = (socket, input_combination)
                # This is a good combination -> Use the looked up op, process next Socket.
                if key in self.call_registry:
                    fetch_list.append(self.call_registry[key])
                    break
        # TODO build feed dict?

        return fetch_list, feed_dict

    def partial_input_build(self, socket):
        """
        Builds one Socket in terms of its incoming and outgoing connections. Returns if we hit a dead end (computation
        that's not complete yet) OR the very end of the graph.
        Recursively calls itself for any encountered Sockets on the way through the Component(s).

        Args:
            socket (Socket): The Socket object to process.
        """
        with tf.variable_scope(socket.scope):
            # Loop through this socket's incoming connections.
            for incoming in socket.incoming_connections:
                # create tf placeholder(s)
                # example: Space (Dict({"a": Discrete(3), "b": Bool(), "c": Continuous()})) connects to Socket ->
                # create inputs[name-of-sock] = dict({"a": tf.placeholder(name="", dtype=int, shape=(3,))})
                socket.update_from_input(incoming, self.op_registry, self.socket_registry)

            for outgoing in socket.outgoing_connections:
                if isinstance(outgoing, Socket):
                    # Outgoing is another Socket -> recurse.
                    self.partial_input_build(outgoing)
                # Outgoing is a Computation -> Add the socket to the computations ("waiting") inputs.
                # ... and maybe build a new op into the graph (via the computation).
                elif isinstance(outgoing, Computation):
                    # We have to specify the device here (only a Computation actually adds something to the graph).
                    assigned_device = socket.device
                    if socket.device:
                        self.assign_device(outgoing, socket, assigned_device)
                    else:
                        # TODO fetch default device?
                        outgoing.update_from_input(socket, self.op_registry, self.socket_registry)
                else:
                    raise YARLError("ERROR: Outgoing connection must be Socket or Computation!")

    def trace_back_sockets(self, trace_set):
        """
        For a given op, returns a list of all Sockets that are required to calculate this op.

        Args:
            trace_set (set): The set of ops to trace-back till the beginning of the Graph.

        Returns: A set of Socket objects that are required to calculate this op.
        """
        # Recursively lookup op in op_registry until we hit a Socket.
        new_trace_set = set()
        for op in trace_set:
            if op not in self.op_registry:
                if op not in self.socket_registry:
                    raise YARLError("ERROR: op {} could not be found in op_registry or socket_registry of model!".
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

