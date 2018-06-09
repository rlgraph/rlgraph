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

from yarl import Specifiable, backend
from yarl.graphs.graph_executor import GraphExecutor
from yarl.utils.input_parsing import parse_execution_spec, parse_update_spec
from yarl.components import  Exploration, PreprocessorStack, NeuralNetwork, Policy, Optimizer, SGDOptimizer
from yarl.graphs import GraphBuilder
from yarl.spaces import Space

import numpy as np
import logging


class Agent(Specifiable):
    """
    Generic agent defining YARL-API operations.
    """
    def __init__(
        self,
        state_space,
        action_space,
        network_spec=None,
        preprocessing_spec=None,
        exploration_spec=None,
        execution_spec=None,
        optimizer_spec=None,
        update_spec=None
    ):
        """
        Generic agent which parses and sanitizes configuration specs.

        Args:
            state_space (Union[dict,Space]): Spec dict for the state Space or a direct Space object.
            action_space (Union[dict,Space]): Spec dict for the action Space or a direct Space object.
            network_spec (Optional[list,NeuralNetwork]): Spec list for a NeuralNetwork Component or the NeuralNetwork
                object itself.
            preprocessing_spec (Optional[list,PreprocessorStack]): The spec list for the different necessary states
                preprocessing steps or a PreprocessorStack object itself.
            exploration_spec (Optional[dict]):
            execution_spec (Optional[dict]):
            optimizer_spec (Optional[dict]):
            update_spec (Optional[dict]): A spec-dict for update (learn) behavior
        """
        self.logger = logging.getLogger(__name__)

        self.state_space = Space.from_spec(state_space).with_batch_rank(False)
        self.logger.info("Parsed state space definition: {}".format(self.state_space))
        self.action_space = Space.from_spec(action_space).with_batch_rank(False)
        self.logger.info("Parsed action space definition: {}".format(self.action_space))

        self.neural_network = None
        self.policy = None

        if network_spec is not None:
            self.neural_network = NeuralNetwork.from_spec(network_spec)
            self.policy = Policy(neural_network=self.neural_network)

        self.preprocessor_stack = PreprocessorStack.from_spec(preprocessing_spec)
        self.exploration = Exploration.from_spec(exploration_spec)
        self.execution_spec = parse_execution_spec(execution_spec)

        # Python-side experience buffer for better performance (may be disabled).
        self.states_buffer = None
        self.actions_buffer = None
        self.internals_buffer = None
        self.reward_buffer = None
        self.terminal_buffer = None
        self.buffer_enabled = self.execution_spec["buffer_enabled"]
        if self.buffer_enabled:
            self.buffer_size = self.execution_spec["buffer_size"]
            self.reset_buffers()
        # Global timesteps counter.
        self.timesteps = 0

        # Create the Agent's optimizer.
        self.optimizer = Optimizer.from_spec(optimizer_spec)
        # Our update-spec dict tells the Agent how (e.g. how often) to update.
        self.update_spec = parse_update_spec(update_spec)

        # Create our GraphBuilder and -Executor.
        self.graph_builder = GraphBuilder(action_space=self.action_space)
        self.graph_executor = GraphExecutor.from_spec(
            backend,
            graph_builder=self.graph_builder,
            execution_spec=self.execution_spec
        )  # type: GraphExecutor

    def reset_buffers(self):
        """
        Initializes buffers for buffered `observe` calls.
        """
        self.states_buffer = list()
        self.actions_buffer = list()
        self.internals_buffer = list()
        self.reward_buffer = list()
        self.terminal_buffer = list()

    def assemble_meta_graph(self):
        """
        Assembles the YARL meta computation graph by combining specified YARL Components.
        Each agent implements this to build its algorithm logic.
        """
        raise NotImplementedError

    def compile_graph(self):
        """
        Asks our GraphExecutor to actually build the Graph.
        """
        self.graph_executor.build()

    def get_action(self, states, deterministic=False):
        """
        Retrieves action(s) for the passed state(s).

        Args:
            states (Union[dict, ndarray]): State dict or array.
            deterministic (bool): If True, no exploration or sampling may be applied
                when retrieving an action.

        Returns: Actions dict.
        """
        raise NotImplementedError

    def observe(self, states, actions, internals, reward, terminal):
        """
        Observes an experience tuple or a batch of experience tuples. Note: If configured,
        first uses buffers and then internally calls _observe_graph() to actually run the computation graph.
        If buffering is disabled, this just routes the call to the respective `_observe_graph()` method of the
        child Agent.

        Args:
            states (Union[dict, ndarray]): States dict or array.
            actions (Union[dict, ndarray]): Actions dict or array containing actions performed for the given state(s).
            internals (Union[list, None]): Internal state(s) returned by agent for the given states.
            reward (float): Scalar reward(s) observed.
            terminal (bool): Boolean indicating terminal.

        """
        if self.buffer_enabled:
            self.states_buffer.append(states)
            self.actions_buffer.append(actions)
            self.internals_buffer.append(internals)
            self.reward_buffer.append(reward)
            self.terminal_buffer.append(terminal)

            # Inserts per episode or when full.
            if len(self.reward_buffer) >= self.buffer_size or terminal:
                self._observe_graph(
                    states=np.asarray(self.states_buffer),
                    actions=np.asarray(self.actions_buffer),
                    internals=np.asarray(self.internals_buffer),
                    reward=np.asarray(self.reward_buffer),
                    terminal=self.terminal_buffer
                )
                self.reset_buffers()
        else:
            self._observe_graph(states, actions, internals, reward, terminal)

    def _observe_graph(self, states, actions, internals, reward, terminal):
        """
        This methods defines the actual call to the computational graph by executing
        the respective graph op via the graph executor. Since this may use varied underlying
        components and inputs, every agent defines which ops it may want to call. The buffered observer
        calls this method to move data into the graph.

        Args:
            states (Union[dict,ndarray]): States dict or array.
            actions (Union[dict,ndarray]): Actions dict or array containing actions performed for the given state(s).
            internals (Union[list,None]): Internal state(s) returned by agent for the given states.
            reward (Union[ndarray,list,float]): Scalar reward(s) observed.
            terminal (Union[list,bool]): Boolean indicating terminal.
        """
        raise NotImplementedError

    def update(self, batch=None):
        """
        Performs an update on the computation graph either via externally experience or
        by sampling from an internal memory.

        Args:
            batch (Optional[dict]): Optional external data batch to use for update. If None, the
                agent should be configured to sample internally.

        Returns:
            Loss value.
        """
        raise NotImplementedError

    def import_observations(self, observations):
        """
        Bulk imports observations, potentially using device pre-fetching. Can be optionally
        implemented by agents requiring pre-training.

        Args:
            observations (dict): Dict or list of observation data.
        """
        pass

    def call_graph_op(self, op, inputs=None):
        """
        Utility method to call any desired operation on the graph, identified via output socket.
        Delegator this call to the YARL graph executor.

        Args:
            op (str): Name of the op, i.e. the name of its output socket on the YARL metagraph.
            inputs (Optional[dict,np.array]): Dict specifying the provided inputs for some in-Sockets (key=in-Socket name,
                values=the values that should go into this Socket (e.g. numpy arrays)).
                Depending on these given inputs, the correct backend-ops can be selected within the given out-Sockets.
                If only one out-Socket is given in `sockets`, and this out-Socket only needs a single in-Socket's data,
                this in-Socket's data may be given here directly.
        Returns:
            any: Result of the op call.
        """
        return self.graph_executor.execute(sockets=op, inputs=inputs)

    def export_graph(self, filename=None):
        """
        Any algorithm defined as a full-graph, as opposed to mixed (mixed Python and graph control flow)
        should be able to export its graph for deployment.

        Args:
            filename (str): Export path. Depending on the backend, different filetypes may be required.
        """
        self.graph_executor.export_graph_definition(filename)

    def store_model(self, path=None, add_timestep=True):
        """
        Store model using the backend's check-pointing mechanism.

        Args:
            path (str): Path to model directory.
            add_timestep (bool): Indiciates if current training step should be appended to
                exported model. If false, may override previous checkpoints.

        """
        self.graph_executor.store_model(path=path, add_timestep=add_timestep)

    def load_model(self, path=None):
        """
        Load model from serialized format.

        Args:
            path (str): Path to checkpoint directory.
        """
        self.graph_executor.load_model(path=path)
