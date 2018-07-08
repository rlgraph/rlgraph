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

import time

from yarl import Specifiable, get_backend
from yarl.graphs.graph_executor import GraphExecutor
from yarl.utils.input_parsing import parse_execution_spec, parse_observe_spec, parse_update_spec, get_optimizer_from_device_strategy
from yarl.components import Component, Exploration, PreprocessorStack, NeuralNetwork
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
        observe_spec=None,
        update_spec=None,
        summary_spec=None,
        name="agent"
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
            exploration_spec (Optional[dict]): The spec-dict to create the Exploration Component.
            execution_spec (Optional[dict,Execution]): The spec-dict specifying execution settings.
            optimizer_spec (Optional[dict,Optimizer]): The spec-dict to create the Optimizer for this Agent.
            observe_spec (Optional[dict]): Spec-dict to specify `Agent.observe()` settings.
            update_spec (Optional[dict]): Spec-dict to specify `Agent.update()` settings.
            summary_spec (Optional[dict]): Spec-dict to specify summary settings.
            name (str): Some name for this Agent object.
        """
        self.name = name
        self.logger = logging.getLogger(__name__)

        self.state_space = Space.from_spec(state_space).with_batch_rank(False)
        self.logger.info("Parsed state space definition: {}".format(self.state_space))
        self.action_space = Space.from_spec(action_space).with_batch_rank(False)
        self.logger.info("Parsed action space definition: {}".format(self.action_space))

        # The agent's core Component.
        self.core_component = Component(name=self.name)

        self.neural_network = None
        self.policy = None

        if network_spec is not None:
            self.neural_network = NeuralNetwork.from_spec(network_spec)

        self.preprocessor = PreprocessorStack.from_spec(preprocessing_spec)
        self.exploration = Exploration.from_spec(exploration_spec)
        self.execution_spec = parse_execution_spec(execution_spec)

        # Python-side experience buffer for better performance (may be disabled).
        self.states_buffer = None
        self.actions_buffer = None
        self.internals_buffer = None
        self.reward_buffer = None
        self.terminal_buffer = None

        self.observe_spec = parse_observe_spec(observe_spec)
        if self.observe_spec["buffer_enabled"]:
            self.reset_buffers()

        # Global timee step counter.
        self.timesteps = 0

        # Create the Agent's optimizer based on optimizer_spec and execution strategy.
        self.optimizer = get_optimizer_from_device_strategy(
            optimizer_spec=optimizer_spec,
            device_strategy=self.execution_spec.get("device_strategy", 'default')
        )

        # Update-spec dict tells the Agent how to update (e.g. memory batch size).
        self.update_spec = parse_update_spec(update_spec)

        # Create our GraphBuilder and -Executor.
        self.graph_builder = GraphBuilder(action_space=self.action_space, summary_spec=summary_spec,
                                          core_component=self.core_component)
        self.graph_executor = GraphExecutor.from_spec(
            get_backend(),
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

    #def define_api_methods(self, *params):
    #    """
    #    Wrapper for the actual `_assemble_meta_graph()` method. Times the meta-graph assembly and logs it.

    #    Args:
    #        params (any): List of parameters to pass on to the Agent's `_assemble_meta_graph` method (after
    #            the core Component).
    #    """
    #    start_time = time.monotonic()
    #    self.logger.info("Start assembly of YARL meta-graph for Agent '{}' ...".format(self.name))
    #    self._assemble_meta_graph(self.graph_builder.core_component, *params)
    #    assembly_time = time.monotonic() - start_time
    #    self.logger.info("YARL meta-graph assembly for Agent '{}' took {} s.".format(self.name, assembly_time))

    def define_api_methods(self, *params):
        """
        Can be used to specify and then `self.define_api_method` the Agent's CoreComponent's API methods.
        Each agent implements this to build its algorithm logic.

        Args:
            params (any): Params to be used freely by child Agent implementations.
        """
        raise NotImplementedError

    def build_graph(self, input_spaces, *args):
        """
        Asks our GraphExecutor to actually build the Graph from the YARL meta-graph.
        """
        self.graph_executor.build(input_spaces, args)

    def get_action(self, states, deterministic=False, return_preprocessed_states=False):
        """
        Returns action(s) for the passed state(s). If `states` is a single state, returns a single action, otherwise,
        returns a batch of actions, where batch-size = number of states passed in.
        Optionally, also returns the preprocessed states.

        Args:
            states (Union[dict,np.ndarray]): State dict/tuple or numpy array.
            deterministic (bool): If True, no exploration or sampling may be applied
                when retrieving an action.
            return_preprocessed_states (bool): Whether to return the preprocessed states as a second
                return value.

        Returns:
            any: Action(s) as dict/tuple/np.ndarray (depending on `self.action_space`).
                Optional: The preprocessed states as a 2nd return value.
        """
        raise NotImplementedError

    def observe(self, states, actions, internals, rewards, terminals):
        """
        Observes an experience tuple or a batch of experience tuples. Note: If configured,
        first uses buffers and then internally calls _observe_graph() to actually run the computation graph.
        If buffering is disabled, this just routes the call to the respective `_observe_graph()` method of the
        child Agent.

        Args:
            states (Union[dict, ndarray]): States dict or array.
            actions (Union[dict, ndarray]): Actions dict or array containing actions performed for the given state(s).
            internals (Union[list]): Internal state(s) returned by agent for the given states.Must be
                empty list if no internals available.
            rewards (float): Scalar reward(s) observed.
            terminals (bool): Boolean indicating terminal.

        """
        batched_states = self.state_space.batched(states)

        # Check for illegal internals.
        if internals is None:
            internals = []

        # Add batch rank?
        if batched_states.ndim == np.asarray(states).ndim + 1:
            states = np.asarray([states])
            actions = np.asarray([actions])
            internals = np.asarray([internals])
            rewards = np.asarray([rewards])
            terminals = np.asarray([terminals])

        if self.observe_spec["buffer_enabled"] is True:
            self.states_buffer.extend(states)
            self.actions_buffer.extend(actions)
            self.internals_buffer.extend(internals)
            self.reward_buffer.extend(rewards)
            self.terminal_buffer.extend(terminals)

            # Inserts per episode or when full.
            if len(self.reward_buffer) >= self.observe_spec["buffer_size"] or terminals:
                self._observe_graph(
                    states=np.asarray(self.states_buffer),
                    actions=np.asarray(self.actions_buffer),
                    internals=np.asarray(self.internals_buffer),
                    rewards=np.asarray(self.reward_buffer),
                    terminals=self.terminal_buffer
                )
                self.reset_buffers()
        else:
            self._observe_graph(states, actions, internals, rewards, terminals)

    def _observe_graph(self, states, actions, internals, rewards, terminals):
        """
        This methods defines the actual call to the computational graph by executing
        the respective graph op via the graph executor. Since this may use varied underlying
        components and api_methods, every agent defines which ops it may want to call. The buffered observer
        calls this method to move data into the graph.

        Args:
            states (Union[dict,ndarray]): States dict or array.
            actions (Union[dict,ndarray]): Actions dict or array containing actions performed for the given state(s).
            internals (Union[list]): Internal state(s) returned by agent for the given states. Must be an empty list
                if no internals available.
            rewards (Union[ndarray,list,float]): Scalar reward(s) observed.
            terminals (Union[list,bool]): Boolean indicating terminal.
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
        Delegate this call to the YARL graph executor.

        Args:
            op (str): Name of the op, i.e. the name of its output socket on the YARL metagraph.
            inputs (Optional[dict,np.array]): Dict specifying the provided api_methods for some in-Sockets (key=in-Socket name,
                values=the values that should go into this Socket (e.g. numpy arrays)).
                Depending on these given api_methods, the correct backend-ops can be selected within the given out-Sockets.
                If only one out-Socket is given in `sockets`, and this out-Socket only needs a single in-Socket's data,
                this in-Socket's data may be given here directly.
        Returns:
            any: Result of the op call.
        """
        return self.graph_executor.execute(api_methods={op: inputs})

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

    def get_weights(self):
        """
        Returns all weights the agents computation graph. Delegates this task to the
        graph executor.

        Returns:
            any: Weights and optionally weight meta data for this model.
        """
        self.graph_executor.get_weights()

    def set_weights(self, weights):
        """
        Sets weights of this agent.  Delegates this task to the
        graph executor.

        Args:
            weights (any): Weights and optionally meta data to update depending on the backend.

        Raises:
            ValueError if weights do not match graph weights in shapes and types.
        """
        self.graph_executor.set_weights(weights=weights)
