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

from collections import defaultdict
import logging
import numpy as np

from rlgraph import get_backend
from rlgraph.components import Component, Exploration, PreprocessorStack, NeuralNetwork, Synchronizable, Policy, \
    Optimizer
from rlgraph.graphs.graph_builder import GraphBuilder
from rlgraph.graphs.graph_executor import GraphExecutor
from rlgraph.spaces.space import Space
from rlgraph.utils.decorators import rlgraph_api
from rlgraph.utils.input_parsing import parse_execution_spec, parse_observe_spec, parse_update_spec
from rlgraph.utils.specifiable import Specifiable


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
        internal_states_space=None,
        action_adapter_spec=None,
        exploration_spec=None,
        execution_spec=None,
        optimizer_spec=None,
        observe_spec=None,
        update_spec=None,
        summary_spec=None,
        saver_spec=None,
        auto_build=True,
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
            internal_states_space (Optional[Union[dict,Space]]): Spec dict for the internal-states Space or a direct
                Space object for the Space(s) of the internal (RNN) states.
            action_adapter_spec (Optional[dict,ActionAdapter]): The spec-dict for the ActionAdapter Component or the
                ActionAdapter object itself.
            exploration_spec (Optional[dict]): The spec-dict to create the Exploration Component.
            execution_spec (Optional[dict,Execution]): The spec-dict specifying execution settings.
            optimizer_spec (Optional[dict,Optimizer]): The spec-dict to create the Optimizer for this Agent.
            observe_spec (Optional[dict]): Spec-dict to specify `Agent.observe()` settings.
            update_spec (Optional[dict]): Spec-dict to specify `Agent.update()` settings.
            summary_spec (Optional[dict]): Spec-dict to specify summary settings.
            saver_spec (Optional[dict]): Spec-dict to specify saver settings.
            auto_build (Optional[bool]): If True (default), immediately builds the graph using the agent's
                graph builder. If false, users must separately call agent.build(). Useful for debugging or analyzing
                components before building.
            name (str): Some name for this Agent object.
        """
        super(Agent, self).__init__()

        self.name = name
        self.auto_build = auto_build
        self.graph_built = False
        self.logger = logging.getLogger(__name__)

        self.state_space = Space.from_spec(state_space).with_batch_rank(False)
        self.logger.info("Parsed state space definition: {}".format(self.state_space))
        self.action_space = Space.from_spec(action_space).with_batch_rank(False)
        self.logger.info("Parsed action space definition: {}".format(self.action_space))

        self.discount = discount

        # The agent's root-Component.
        self.root_component = Component(name=self.name)

        # Define the input-Spaces:
        # Tag the input-Space to `self.set_policy_weights` as equal to whatever the variables-Space will be for
        # the Agent's policy Component.
        self.input_spaces = dict(
            states=self.state_space.with_batch_rank(),
        )

        # Construct the Preprocessor.
        self.preprocessor = PreprocessorStack.from_spec(preprocessing_spec)
        self.preprocessed_state_space = self.preprocessor.get_preprocessed_space(self.state_space)
        self.preprocessing_required = preprocessing_spec is not None and len(preprocessing_spec) > 1
        if self.preprocessing_required:
            self.logger.info("Preprocessing required.")
            self.logger.info("Parsed preprocessed-state space definition: {}".format(self.preprocessed_state_space))
        else:
            self.logger.info("No preprocessing required.")

        # Construct the Policy network.
        self.neural_network = None
        if network_spec is not None:
            self.neural_network = NeuralNetwork.from_spec(network_spec)
        self.action_adapter_spec = action_adapter_spec

        self.internal_states_space = internal_states_space

        # An object implementing the loss function interface is only strictly needed
        # if automatic device strategies like multi-gpu are enabled. This is because
        # the device strategy needs to know the name of the loss function to infer the appropriate
        # operations.
        self.loss_function = None

        # The action adapter mapping raw NN output to (shaped) actions.
        action_adapter_dict = dict(action_space=self.action_space)
        if self.action_adapter_spec is None:
            self.action_adapter_spec = action_adapter_dict
        else:
            self.action_adapter_spec.update(action_adapter_dict)

        # The behavioral policy of the algorithm. Also the one that gets updated.
        self.policy = Policy(
            network_spec=self.neural_network,
            action_adapter_spec=self.action_adapter_spec
        )

        self.exploration = Exploration.from_spec(exploration_spec)
        self.execution_spec = parse_execution_spec(execution_spec)

        # Python-side experience buffer for better performance (may be disabled).
        self.default_env = "env_0"
        self.states_buffer = defaultdict(list)
        self.actions_buffer = defaultdict(list)
        self.internals_buffer = defaultdict(list)
        self.rewards_buffer = defaultdict(list)
        self.next_states_buffer = defaultdict(list)
        self.terminals_buffer = defaultdict(list)

        self.observe_spec = parse_observe_spec(observe_spec)
        if self.observe_spec["buffer_enabled"]:
            self.reset_env_buffers()

        # Global time step counter.
        self.timesteps = 0

        # Create the Agent's optimizer based on optimizer_spec and execution strategy.
        self.optimizer = None
        if optimizer_spec is not None:
            self.optimizer = Optimizer.from_spec(optimizer_spec)  #get_optimizer_from_device_strategy(
                #optimizer_spec, self.execution_spec.get("device_strategy", 'default')
        # Update-spec dict tells the Agent how to update (e.g. memory batch size).
        self.update_spec = parse_update_spec(update_spec)

        # Create our GraphBuilder and -Executor.
        self.graph_builder = GraphBuilder(action_space=self.action_space, summary_spec=summary_spec)
        self.graph_executor = GraphExecutor.from_spec(
            get_backend(),
            graph_builder=self.graph_builder,
            execution_spec=self.execution_spec,
            saver_spec=saver_spec
        )  # type: GraphExecutor

    def reset_env_buffers(self, env_id=None):
        """
        Resets an environment buffer for buffered `observe` calls.

        Args:
            env_id (Optional[str]): Environment id to reset. Defaults to a default environment if None provided.
        """
        if env_id is None:
            env_id = self.default_env
        self.states_buffer[env_id] = []
        self.actions_buffer[env_id] = []
        self.internals_buffer[env_id] = []
        self.rewards_buffer[env_id] = []
        self.next_states_buffer[env_id] = []
        self.terminals_buffer[env_id] = []

    # TODO optimizer scope missing?
    def define_graph_api(self, policy_scope, pre_processor_scope, *params):
        """
        Can be used to specify and then `self.define_api_method` the Agent's CoreComponent's API methods.
        Each agent implements this to build its algorithm logic.

        Args:
            policy_scope (str): The global scope of the Policy within the Agent.
            pre_processor_scope (str): The global scope of the PreprocessorStack within the Agent.
            params (any): Params to be used freely by child Agent implementations.
        """
        # Done by default.
        # TODO: Move this to ctor as this belongs to the init phase and doesn't really have to do with API-methods.
        self.policy.add_components(Synchronizable(), expose_apis="sync")

        # Add api methods for syncing.
        @rlgraph_api(component=self.root_component)
        def get_policy_weights(self):
            policy = self.get_sub_component_by_name(policy_scope)
            return policy._variables()

        @rlgraph_api(component=self.root_component, must_be_complete=False)
        def set_policy_weights(self, weights):
            policy = self.get_sub_component_by_name(policy_scope)
            return policy.sync(weights)

        # To pre-process external data if needed.
        @rlgraph_api(component=self.root_component)
        def preprocess_states(self, states):
            preprocessor_stack = self.get_sub_component_by_name(pre_processor_scope)
            preprocessed_states = preprocessor_stack.preprocess(states)
            return preprocessed_states

    def _build_graph(self, root_components, input_spaces, **kwargs):
        """
        Builds the internal graph from the RLGraph meta-graph via the graph executor..
        """
        return self.graph_executor.build(root_components, input_spaces, **kwargs)

    def build(self, build_options=None):
        """
        Builds this agent. This method call only be called if the agent parameter "auto_build"
        was set to False.

        Args:
            build_options (Optional[dict]): Optional build options, see build doc.
        """
        assert not self.graph_built,\
            "ERROR: Attempting to build agent which has already been built. Ensure auto_build parameter is set to " \
            "False (was {}), and method has not been called twice".format(self.auto_build)

        # TODO let agent have a list of root-components
        return self._build_graph(
            [self.root_component], self.input_spaces, optimizer=self.optimizer,
            build_options=build_options, batch_size=self.update_spec["batch_size"]
        )

    def preprocess_states(self, states):
        """
        Applies the agent's preprocessor to one or more states, e.g. to preprocess external data
        before inserting to memory without acting. Returns identity if no preprocessor defined.

        Args:
            states (np.array): State(s) to preprocess.

        Returns:
            np.array: Preprocessed states.
        """
        if self.preprocessing_required:
            return self.call_api_method("preprocess_states", states)
        else:
            # Return identity.
            return states

    def get_action(self, states, internals=None, use_exploration=True, apply_preprocessing=True, extra_returns=None):
        """
        Returns action(s) for the passed state(s). If `states` is a single state, returns a single action, otherwise,
        returns a batch of actions, where batch-size = number of states passed in.

        Args:
            states (Union[dict,np.ndarray]): States dict/tuple or numpy array.
            internals (Union[dict,np.ndarray]): Internal states dict/tuple or numpy array.

            use_exploration (bool): If False, no exploration or sampling may be applied
                when retrieving an action.

            apply_preprocessing (bool): If True, apply any state preprocessors configured to the action. Set to
                false if all pre-processing is handled externally both for acting and updating.

            extra_returns (Optional[Set[str]]): Optional set of Agent-specific strings for additional return
                values (besides the actions). All Agents must support "preprocessed_states".

        Returns:
            any: Action(s) as dict/tuple/np.ndarray (depending on `self.action_space`).
                Optional: The preprocessed states as a 2nd return value.
        """
        raise NotImplementedError

    def observe(self, preprocessed_states, actions, internals, rewards, next_states, terminals, env_id=None):
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
            next_states (Union[dict, ndarray]): Preprocessed next states dict or array.

            env_id (Optional[str]): Environment id to observe for. When using vectorized execution and
                buffering, using environment ids is necessary to ensure correct trajectories are inserted.
                See `SingleThreadedWorker` for example usage.
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
            # Also batch next_states (or already done?).
            if next_states.ndim == preprocessed_states.ndim - 1:
                next_states = np.asarray([next_states])

        if self.observe_spec["buffer_enabled"] is True:
            if env_id is None:
                env_id = self.default_env

            self.states_buffer[env_id].extend(preprocessed_states)
            self.actions_buffer[env_id].extend(actions)
            self.internals_buffer[env_id].extend(internals)
            self.rewards_buffer[env_id].extend(rewards)
            self.next_states_buffer[env_id].extend(next_states)
            self.terminals_buffer[env_id].extend(terminals)

            buffer_is_full = len(self.rewards_buffer[env_id]) >= self.observe_spec["buffer_size"]

            # If the buffer (per environment) is full OR the episode was aborted:
            # Change terminal of last record artificially to True, insert and flush the buffer.
            if buffer_is_full or self.terminals_buffer[env_id][-1]:
                self.terminals_buffer[env_id][-1] = True

                # TODO: Apply n-step post-processing if necessary.
                # if self.observe_spec["n_step"] > 1:
                #    pass

                self._observe_graph(
                    preprocessed_states=np.asarray(self.states_buffer[env_id]),
                    actions=np.asarray(self.actions_buffer[env_id]),
                    internals=np.asarray(self.internals_buffer[env_id]),
                    rewards=np.asarray(self.rewards_buffer[env_id]),
                    next_states=np.asarray(self.next_states_buffer[env_id]),
                    terminals=np.asarray(self.terminals_buffer[env_id])
                )
                self.reset_env_buffers(env_id)
        else:
            self._observe_graph(preprocessed_states, actions, internals, rewards, next_states, terminals)

    def _observe_graph(self, preprocessed_states, actions, internals, rewards, next_states, terminals):
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
            next_states (Union[dict, ndarray]): Preprocessed next states dict or array.
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

    def terminate(self):
        """
        Terminates the Agent, so it will no longer be usable.
        Things that need to be cleaned up should be placed into this function, e.g. closing sessions
        and other open connections.
        """
        self.graph_executor.terminate()

    def call_api_method(self, op, inputs=None, return_ops=None):
        """
        Utility method to call any desired api method on the graph, identified via output socket.
        Delegate this call to the RLGraph graph executor.

        Args:
            op (str): Name of the api method.

            inputs (Optional[dict,np.array]): Dict specifying the provided api_methods for (key=input space name,
                values=the values that should go into this space (e.g. numpy arrays)).
        Returns:
            any: Result of the op call.
        """
        return self.graph_executor.execute((op, inputs, return_ops))

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

            add_timestep (bool): Indiciates if current training step should be appended to exported model.
                If false, may override previous checkpoints.
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

