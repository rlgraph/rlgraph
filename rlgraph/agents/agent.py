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

from rlgraph import Specifiable, get_backend
from rlgraph.graphs.graph_executor import GraphExecutor
from rlgraph.utils.input_parsing import parse_execution_spec, parse_observe_spec, parse_update_spec,\
    get_optimizer_from_device_strategy
from rlgraph.components import Component, Exploration, PreprocessorStack, NeuralNetwork, Synchronizable, Policy
from rlgraph.graphs import GraphBuilder
from rlgraph.spaces import Space

import numpy as np
import logging


class Agent(Specifiable):
    """
    Generic agent defining RLGraph-API operations and parses and sanitizes configuration specs.
    """
    def __init__(
        self,
        state_space,
        action_space,
        discount=0.98,
        preprocessing_spec=None,
        network_spec=None,
        action_adapter_spec=None,
        exploration_spec=None,
        execution_spec=None,
        optimizer_spec=None,
        observe_spec=None,
        update_spec=None,
        summary_spec=None,
        name="agent"
    ):
        """
        Args:
            state_space (Union[dict,Space]): Spec dict for the state Space or a direct Space object.
            action_space (Union[dict,Space]): Spec dict for the action Space or a direct Space object.
            preprocessing_spec (Optional[list,PreprocessorStack]): The spec list for the different necessary states
                preprocessing steps or a PreprocessorStack object itself.
            discount (float): The discount factor (gamma).
            network_spec (Optional[list,NeuralNetwork]): Spec list for a NeuralNetwork Component or the NeuralNetwork
                object itself.
            action_adapter_spec (Optional[dict,ActionAdapter]): The spec-dict for the ActionAdapter Component or the
                ActionAdapter object itself.
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

        self.discount = discount

        # The agent's core Component.
        self.core_component = Component(name=self.name)

        # Define the input-Spaces:
        # Tag the input-Space to `self.set_policy_weights` as equal to whatever the variables-Space will be for
        # the Agent's policy Component.
        self.input_spaces = dict(
            set_policy_weights="variables:policy"
        )

        # Construct the Preprocessor.
        self.preprocessor = PreprocessorStack.from_spec(preprocessing_spec)
        self.preprocessed_state_space = self.preprocessor.get_preprocessed_space(self.state_space)
        self.logger.info("Parsed preprocessed-state space definition: {}".format(self.preprocessed_state_space))

        # Construct the Policy network.
        self.neural_network = None
        if network_spec is not None:
            self.neural_network = NeuralNetwork.from_spec(network_spec)
        self.action_adapter_spec = action_adapter_spec

        # The behavioral policy of the algorithm. Also the one that gets updated.
        action_adapter_dict = dict(action_space=self.action_space)
        if self.action_adapter_spec is None:
            self.action_adapter_spec = action_adapter_dict
        else:
            self.action_adapter_spec.update(action_adapter_dict)
        self.policy = Policy(
            neural_network=self.neural_network,
            action_adapter_spec=self.action_adapter_spec
        )

        self.exploration = Exploration.from_spec(exploration_spec)
        self.execution_spec = parse_execution_spec(execution_spec)

        # Python-side experience buffer for better performance (may be disabled).
        self.states_buffer = None
        self.actions_buffer = None
        self.internals_buffer = None
        self.rewards_buffer = None
        self.terminals_buffer = None

        self.observe_spec = parse_observe_spec(observe_spec)
        if self.observe_spec["buffer_enabled"]:
            self.reset_buffers()

        # Global time step counter.
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
        self.rewards_buffer = list()
        self.terminals_buffer = list()

    def define_api_methods(self, *params):
        """
        Can be used to specify and then `self.define_api_method` the Agent's CoreComponent's API methods.
        Each agent implements this to build its algorithm logic.

        Args:
            params (any): Params to be used freely by child Agent implementations.
        """
        self.policy.add_components(Synchronizable(), expose_apis="sync")

        # Add api methods for syncing.
        def get_policy_weights(self_):
            return self_.call(self.policy._variables)

        self.core_component.define_api_method("get_policy_weights", get_policy_weights)

        def set_policy_weights(self_, weights):
            return self_.call(self.policy.sync, weights)

        self.core_component.define_api_method("set_policy_weights", set_policy_weights, must_be_complete=False)

    def build_graph(self, input_spaces, *args):
        """
        Asks our GraphExecutor to actually build the Graph from the RLGraph meta-graph.
        """
        self.graph_executor.build(input_spaces, *args)

    def get_action(self, states, internals=None, use_exploration=True, extra_returns=None):
        """
        Returns action(s) for the passed state(s). If `states` is a single state, returns a single action, otherwise,
        returns a batch of actions, where batch-size = number of states passed in.

        Args:
            states (Union[dict,np.ndarray]): States dict/tuple or numpy array.
            internals (Union[dict,np.ndarray]): Internal states dict/tuple or numpy array.
            use_exploration (bool): If False, no exploration or sampling may be applied
                when retrieving an action.
            extra_returns (Optional[Set[str]]): Optional set of Agent-specific strings for additional return
                values (besides the actions). All Agents must support "preprocessed_states".

        Returns:
            any: Action(s) as dict/tuple/np.ndarray (depending on `self.action_space`).
                Optional: The preprocessed states as a 2nd return value.
        """
        raise NotImplementedError

    def observe(self, preprocessed_states, actions, internals, rewards, terminals):
        """
        Observes an experience tuple or a batch of experience tuples. Note: If configured,
        first uses buffers and then internally calls _observe_graph() to actually run the computation graph.
        If buffering is disabled, this just routes the call to the respective `_observe_graph()` method of the
        child Agent.

        Args:
            preprocessed_states (Union[dict, ndarray]): Preprocessed states dict or array.
            actions (Union[dict, ndarray]): Actions dict or array containing actions performed for the given state(s).
            internals (Union[list]): Internal state(s) returned by agent for the given states.Must be
                empty list if no internals available.
            rewards (float): Scalar reward(s) observed.
            terminals (bool): Boolean indicating terminal.
        """
        batched_states = self.preprocessed_state_space.force_batch(preprocessed_states)

        # Check for illegal internals.
        if internals is None:
            internals = []

        # Add batch rank?
        if batched_states.ndim == np.asarray(preprocessed_states).ndim + 1:
            preprocessed_states = np.asarray([preprocessed_states])
            actions = np.asarray([actions])
            internals = np.asarray([internals])
            rewards = np.asarray([rewards])
            terminals = np.asarray([terminals])

        if self.observe_spec["buffer_enabled"] is True:
            self.states_buffer.extend(preprocessed_states)
            self.actions_buffer.extend(actions)
            self.internals_buffer.extend(internals)
            self.rewards_buffer.extend(rewards)
            self.terminals_buffer.extend(terminals)

            buffer_is_full = len(self.rewards_buffer) >= self.observe_spec["buffer_size"]
            # If the buffer is full OR the episode was aborted:
            # Change terminal of last record artificially to True, insert and flush the buffer.
            if buffer_is_full or self.terminals_buffer[-1]:
                self.terminals_buffer[-1] = True
                self._observe_graph(
                    preprocessed_states=np.asarray(self.states_buffer),
                    actions=np.asarray(self.actions_buffer),
                    internals=np.asarray(self.internals_buffer),
                    rewards=np.asarray(self.rewards_buffer),
                    terminals=np.asarray(self.terminals_buffer)
                )
                self.reset_buffers()
        else:
            self._observe_graph(preprocessed_states, actions, internals, rewards, terminals)

    def _observe_graph(self, preprocessed_states, actions, internals, rewards, terminals):
        """
        This methods defines the actual call to the computational graph by executing
        the respective graph op via the graph executor. Since this may use varied underlying
        components and api_methods, every agent defines which ops it may want to call. The buffered observer
        calls this method to move data into the graph.

        Args:
            preprocessed_states (Union[dict,ndarray]): Preprocessed states dict or array.
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
            float: The loss value calculated in this update.
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

    def reset(self):
        """
        Must be implemented to define some reset behavior (before starting a new episode).
        This could include resetting the preprocessor and other Components.
        """
        pass  # optional

    def call_graph_op(self, op, inputs=None):
        """
        Utility method to call any desired operation on the graph, identified via output socket.
        Delegate this call to the RLGraph graph executor.

        Args:
            op (str): Name of the op, i.e. the name of its output socket on the RLGraph metagraph.
            inputs (Optional[dict,np.array]): Dict specifying the provided api_methods for some in-Sockets (key=in-Socket name,
                values=the values that should go into this Socket (e.g. numpy arrays)).
                Depending on these given api_methods, the correct backend-ops can be selected within the given out-Sockets.
                If only one out-Socket is given in `sockets`, and this out-Socket only needs a single in-Socket's data,
                this in-Socket's data may be given here directly.
        Returns:
            any: Result of the op call.
        """
        return self.graph_executor.execute((op, inputs))

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

    def get_policy_weights(self):
        """
        Returns all weights relevant for the agent's policy for syncing purposes.

        Returns:
            any: Weights and optionally weight meta data for this model.
        """
        return dict(self.graph_executor.execute("get_policy_weights"))

    def set_policy_weights(self, weights):
        """
        Sets policy weights of this agent, e.g. for external syncing purporses.

        Args:
            weights (any): Weights and optionally meta data to update depending on the backend.

        Raises:
            ValueError if weights do not match graph weights in shapes and types.
        """
        return self.graph_executor.execute(("set_policy_weights", weights))

