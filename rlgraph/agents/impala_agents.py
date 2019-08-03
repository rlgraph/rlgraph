# Copyright 2018/2019 The RLgraph authors. All Rights Reserved.
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

from __future__ import absolute_import, division, print_function

import copy
from collections import defaultdict
from functools import partial
import logging

import numpy as np

from rlgraph import get_backend
from rlgraph.components import Component, Exploration, PreprocessorStack, Synchronizable, Policy, Optimizer
from rlgraph.graphs.graph_builder import GraphBuilder
from rlgraph.graphs.graph_executor import GraphExecutor
from rlgraph.spaces import Space, ContainerSpace, FloatBox, Dict, Tuple
from rlgraph.utils.input_parsing import parse_execution_spec, parse_observe_spec, parse_update_spec, \
    parse_value_function_spec
from rlgraph.utils.ops import flatten_op
from rlgraph.utils.specifiable import Specifiable
from rlgraph.components.common.container_merger import ContainerMerger
from rlgraph.components.common.environment_stepper import EnvironmentStepper
from rlgraph.components.common.slice import Slice
from rlgraph.components.common.staging_area import StagingArea
from rlgraph.components.layers.preprocessing.container_splitter import ContainerSplitter
from rlgraph.components.layers.preprocessing.reshape import ReShape
from rlgraph.components.layers.preprocessing.transpose import Transpose
from rlgraph.components.loss_functions.impala_loss_function import IMPALALossFunction
from rlgraph.components.memories.fifo_queue import FIFOQueue
from rlgraph.components.memories.queue_runner import QueueRunner
from rlgraph.components.neural_networks.actor_component import ActorComponent
from rlgraph.components.policies.dynamic_batching_policy import DynamicBatchingPolicy
from rlgraph.utils import RLGraphError
from rlgraph.utils.decorators import rlgraph_api, graph_fn
from rlgraph.utils.util import default_dict

if get_backend() == "tf":
    import tensorflow as tf


# TODO: This older-version base-Agent class is now only used  by IMPALA and will be retired once IMPALA
# has been translated to the new Agent API.
class OldBaseAgent(Specifiable):
    """
    Generic agent defining RLGraph-API operations and parses and sanitizes configuration specs.
    """
    def __init__(self, state_space, action_space, discount=0.98,
                 preprocessing_spec=None, network_spec=None, internal_states_space=None,
                 policy_spec=None, value_function_spec=None,
                 exploration_spec=None, execution_spec=None, optimizer_spec=None, value_function_optimizer_spec=None,
                 observe_spec=None, update_spec=None, max_timesteps=None,
                 summary_spec=None, saver_spec=None, auto_build=True, name="agent"):
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

            policy_spec (Optional[dict]): An optional dict for further kwargs passing into the Policy c'tor.
            value_function_spec (list, dict, ValueFunction): Neural network specification for baseline or instance
                of ValueFunction.

            exploration_spec (Optional[dict]): The spec-dict to create the Exploration Component.
            execution_spec (Optional[dict,Execution]): The spec-dict specifying execution settings.
            optimizer_spec (Optional[dict,Optimizer]): The spec-dict to create the Optimizer for this Agent.

            value_function_optimizer_spec (dict): Optimizer config for value function optimizer. If None, the optimizer
                spec for the policy is used (same learning rate and optimizer type).

            observe_spec (Optional[dict]): Spec-dict to specify `Agent.observe()` settings.
            update_spec (Optional[dict]): Spec-dict to specify `Agent.update()` settings.
            max_timesteps (Optional[int]): An optional max timesteps hint for Workers.
            summary_spec (Optional[dict]): Spec-dict to specify summary settings.
            saver_spec (Optional[dict]): Spec-dict to specify saver settings.

            auto_build (Optional[bool]): If True (default), immediately builds the graph using the agent's
                graph builder. If false, users must separately call agent.build(). Useful for debugging or analyzing
                components before building.

            name (str): Some name for this Agent object.
        """
        super(OldBaseAgent, self).__init__()
        self.name = name
        self.auto_build = auto_build
        self.graph_built = False
        self.logger = logging.getLogger(__name__)

        self.state_space = Space.from_spec(state_space).with_batch_rank(False)
        self.flat_state_space = self.state_space.flatten(scope_separator_at_start=False)\
            if isinstance(self.state_space, ContainerSpace) else None
        self.logger.info("Parsed state space definition: {}".format(self.state_space))
        self.action_space = Space.from_spec(action_space).with_batch_rank(False)
        self.flat_action_space = self.action_space.flatten() if isinstance(self.action_space, ContainerSpace) else None
        self.logger.info("Parsed action space definition: {}".format(self.action_space))

        self.discount = discount
        self.build_options = {}

        # The agent's root-Component.
        self.root_component = Component(name=self.name, nesting_level=0)

        # Define the input-Spaces:
        # Tag the input-Space to `self.set_weights` as equal to whatever the variables-Space will be for
        # the Agent's policy Component.
        self.input_spaces = dict(
            states=self.state_space.with_batch_rank(),
            time_percentage=float,
            #increment=int,
            #episode_reward=float
        )

        # Construct the Preprocessor.
        self.preprocessor = PreprocessorStack.from_spec(preprocessing_spec)
        self.preprocessed_state_space = self.preprocessor.get_preprocessed_space(self.state_space)
        self.preprocessing_required = preprocessing_spec is not None and len(preprocessing_spec) > 0
        if self.preprocessing_required:
            self.logger.info("Preprocessing required.")
            self.logger.info("Parsed preprocessed-state space definition: {}".format(self.preprocessed_state_space))
        else:
            self.logger.info("No preprocessing required.")

        # Construct the Policy network.
        policy_spec = policy_spec or {}
        if "network_spec" not in policy_spec:
            policy_spec["network_spec"] = network_spec
        if "action_space" not in policy_spec:
            policy_spec["action_space"] = self.action_space
        self.policy_spec = policy_spec
        # The behavioral policy of the algorithm. Also the one that gets updated.
        self.policy = Policy.from_spec(self.policy_spec)
        # Done by default.
        self.policy.add_components(Synchronizable(), expose_apis="sync")

        # Create non-shared baseline network.
        self.value_function = parse_value_function_spec(value_function_spec, skip_obsoleted_error=True)
        # TODO move this to specific agents.
        if self.value_function is not None:
            self.vars_merger = ContainerMerger("policy", "vf", scope="variable-dict-merger")
            self.vars_splitter = ContainerSplitter("policy", "vf", scope="variable-container-splitter")
        else:
            self.vars_merger = ContainerMerger("policy", scope="variable-dict-merger")
            self.vars_splitter = ContainerSplitter("policy", scope="variable-container-splitter")
        self.internal_states_space = Space.from_spec(internal_states_space)

        # An object implementing the loss function interface is only strictly needed
        # if automatic device strategies like multi-gpu are enabled. This is because
        # the device strategy needs to know the name of the loss function to infer the appropriate
        # operations.
        self.loss_function = None

        self.exploration = Exploration.from_spec(exploration_spec)  # TODO: Move this to DQN/DQFN. PG's don't use it.
        self.execution_spec = parse_execution_spec(execution_spec)

        # Python-side experience buffer for better performance (may be disabled).
        self.default_env = "env_0"

        def factory_(i):
            if i == 0:
                return []
            return tuple([[] for _ in range(i)])

        self.states_buffer = defaultdict(partial(factory_, len(self.flat_state_space or [])))
        self.actions_buffer = defaultdict(partial(factory_, len(self.flat_action_space or [])))
        self.internals_buffer = defaultdict(list)
        self.rewards_buffer = defaultdict(list)
        self.next_states_buffer = defaultdict(partial(factory_, len(self.flat_state_space or [])))
        self.terminals_buffer = defaultdict(list)

        self.observe_spec = parse_observe_spec(observe_spec, skip_obsoleted_error=True)

        # Global time step counter.
        self.timesteps = 0
        # Global updates counter.
        self.num_updates = 0
        # An optional maximum timestep value to use by Workers to figure out `time_percentage` in calls to
        # `update()` and `get_action()`.
        self.max_timesteps = max_timesteps

        # Create the Agent's optimizer based on optimizer_spec and execution strategy.
        self.optimizer = None
        if optimizer_spec is not None:
            # Save spec in case agent needs to create more optimizers e.g. for baseline.
            self.optimizer_spec = optimizer_spec
            self.optimizer = Optimizer.from_spec(optimizer_spec)

        self.value_function_optimizer = None
        if self.value_function is not None:
            if value_function_optimizer_spec is None:
                vf_optimizer_spec = self.optimizer_spec
            else:
                vf_optimizer_spec = value_function_optimizer_spec
            vf_optimizer_spec["scope"] = "value-function-optimizer"
            self.value_function_optimizer = Optimizer.from_spec(vf_optimizer_spec)

        # Update-spec dict tells the Agent how to update (e.g. memory batch size).
        self.update_spec = parse_update_spec(update_spec, skip_obsoleted_error=True)

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
        del self.states_buffer[env_id]  # = ([] for _ in range(len(self.flat_state_space)))
        del self.actions_buffer[env_id]  # = ([] for _ in range(len(self.flat_action_space)))
        del self.internals_buffer[env_id]  # = []
        del self.rewards_buffer[env_id]  # = []
        del self.next_states_buffer[env_id]  # = ([] for _ in range(len(self.flat_state_space)))
        del self.terminals_buffer[env_id]  # = []

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
        if build_options is not None:
            self.build_options.update(build_options)
        assert not self.graph_built, \
            "ERROR: Attempting to build agent which has already been built. Ensure auto_build parameter is set to " \
            "False (was {}), and method has not been called twice".format(self.auto_build)

        # TODO let agent have a list of root-components
        return self._build_graph(
            [self.root_component], self.input_spaces,
            build_options=self.build_options, batch_size=self.update_spec["batch_size"]
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

    def get_action(self, states, internals=None, use_exploration=True, apply_preprocessing=True, extra_returns=None,
                   time_percentage=None):
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

    def observe(self, preprocessed_states, actions, internals, rewards, next_states, terminals, env_id=None,
                batched=False):
        """
        Observes an experience tuple or a batch of experience tuples. Note: If configured,
        first uses buffers and then internally calls _observe_graph() to actually run the computation graph.
        If buffering is disabled, this just routes the call to the respective `_observe_graph()` method of the
        child Agent.

        Args:
            preprocessed_states (Union[dict,ndarray]): Preprocessed states dict or array.
            actions (Union[dict,ndarray]): Actions dict or array containing actions performed for the given state(s).

            internals (Optional[list]): Internal state(s) returned by agent for the given states.Must be
                empty list if no internals available.

            rewards (Union[float,List[float]]): Scalar reward(s) observed.
            terminals (Union[bool,List[bool]]): Boolean indicating terminal.
            next_states (Union[dict,ndarray]): Preprocessed next states dict or array.

            env_id (Optional[str]): Environment id to observe for. When using vectorized execution and
                buffering, using environment ids is necessary to ensure correct trajectories are inserted.
                See `SingleThreadedWorker` for example usage.

            batched (bool): Whether given data (states, actions, etc..) is already batched or not.
        """
        # Check for illegal internals.
        if internals is None:
            internals = []

        if self.observe_spec["buffer_enabled"] is True:
            if env_id is None:
                env_id = self.default_env

            # If data is already batched, just have to extend our buffer lists.
            if batched:
                if self.flat_state_space is not None:
                    for i, flat_key in enumerate(self.flat_state_space.keys()):
                        self.states_buffer[env_id][i].extend(preprocessed_states[flat_key])
                        self.next_states_buffer[env_id][i].extend(next_states[flat_key])
                else:
                    self.states_buffer[env_id].extend(preprocessed_states)
                    self.next_states_buffer[env_id].extend(next_states)
                if self.flat_action_space is not None:
                    flat_action = flatten_op(actions)
                    for i, flat_key in enumerate(self.flat_action_space.keys()):
                        self.actions_buffer[env_id][i].append(flat_action[flat_key])
                else:
                    self.actions_buffer[env_id].extend(actions)
                self.internals_buffer[env_id].extend(internals)
                self.rewards_buffer[env_id].extend(rewards)
                self.terminals_buffer[env_id].extend(terminals)
            # Data is not batched, append single items (without creating new lists first!) to buffer lists.
            else:
                if self.flat_state_space is not None:
                    for i, flat_key in enumerate(self.flat_state_space.keys()):
                        self.states_buffer[env_id][i].append(preprocessed_states[flat_key])
                        self.next_states_buffer[env_id][i].append(next_states[flat_key])
                else:
                    self.states_buffer[env_id].append(preprocessed_states)
                    self.next_states_buffer[env_id].append(next_states)
                if self.flat_action_space is not None:
                    flat_action = flatten_op(actions)
                    for i, flat_key in enumerate(self.flat_action_space.keys()):
                        self.actions_buffer[env_id][i].append(flat_action[flat_key])
                else:
                    self.actions_buffer[env_id].append(actions)
                self.internals_buffer[env_id].append(internals)
                self.rewards_buffer[env_id].append(rewards)
                self.terminals_buffer[env_id].append(terminals)

            buffer_is_full = len(self.rewards_buffer[env_id]) >= self.observe_spec["buffer_size"]

            # If the buffer (per environment) is full OR the episode was aborted:
            # Change terminal of last record artificially to True (also give warning "buffer too small"),
            # insert and flush the buffer.
            if buffer_is_full or self.terminals_buffer[env_id][-1]:
                # Warn if full and last terminal is False.
                if buffer_is_full and not self.terminals_buffer[env_id][-1]:
                    self.logger.warning(
                        "Buffer of size {} of Agent '{}' may be too small! Had to add artificial terminal=True "
                        "to end.".format(self.observe_spec["buffer_size"], self)
                    )
                    self.terminals_buffer[env_id][-1] = True

                # TODO: Apply n-step post-processing if necessary.
                if self.flat_state_space is not None:
                    states_ = {}
                    next_states_ = {}
                    for i, key in enumerate(self.flat_state_space.keys()):
                        states_[key] = np.asarray(self.states_buffer[env_id][i])
                        next_states_[key] = np.asarray(self.next_states_buffer[env_id][i])
                        # Squeeze, but do not squeeze (1,) to ().
                        if len(states_[key]) > 1:
                            states_[key] = np.squeeze(states_[key])
                            next_states_[key] = np.squeeze(next_states_[key])
                        else:
                            states_[key] = np.reshape(states_[key], (1,))
                            next_states_[key] = np.reshape(next_states_[key], (1,))
                else:
                    states_ = np.asarray(self.states_buffer[env_id])
                    next_states_ = np.asarray(self.next_states_buffer[env_id])

                if self.flat_action_space is not None:
                    actions_ = {}
                    for i, key in enumerate(self.flat_action_space.keys()):
                        actions_[key] = np.asarray(self.actions_buffer[env_id][i])
                        # Squeeze, but do not squeeze (1,) to ().
                        if len(actions_[key]) > 1:
                            actions_[key] = np.squeeze(actions_[key])
                        else:
                            actions_[key] = np.reshape(actions_[key], (1,))
                else:
                    actions_ = np.asarray(self.actions_buffer[env_id])

                self._write_rewards_summary(
                    rewards=self.rewards_buffer[env_id],  # No need to be converted to np
                    terminals=self.terminals_buffer[env_id],
                    env_id=env_id
                )

                self._observe_graph(
                    preprocessed_states=states_,
                    actions=actions_,
                    internals=np.asarray(self.internals_buffer[env_id]),
                    rewards=np.asarray(self.rewards_buffer[env_id]),
                    next_states=next_states_,
                    terminals=np.asarray(self.terminals_buffer[env_id])
                )
                self.reset_env_buffers(env_id)
        else:
            if not batched:
                preprocessed_states, _ = self.preprocessed_state_space.force_batch(preprocessed_states)
                next_states, _ = self.preprocessed_state_space.force_batch(next_states)
                actions, _ = self.action_space.force_batch(actions)
                rewards = [rewards]
                terminals = [terminals]

            self._write_rewards_summary(
                rewards=rewards,  # No need to be converted to np
                terminals=terminals,
                env_id=env_id
            )

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

    def _write_rewards_summary(self, rewards, terminals, env_id):
        """
        Writes summary for the observed rewards.

        Args:
            rewards (float): The observed rewards.
            terminals (bool): The observed episode terminal states.
            env_id (str): The id of the environment.
        """
        # TODO: 1. handle env_id
        # TODO: 2. control the level of verbosity
        # TODO: 3. can we reduce the graph interactions?
        ret = self.graph_executor.execute(
            "get_episode_reward", "get_global_timestep"
        )
        episode_reward = ret["get_episode_reward"]
        timestep = ret["get_global_timestep"]
        for i in range(len(rewards)):
            summary = tf.Summary(value=[
                tf.Summary.Value(tag="reward/raw_reward", simple_value=rewards[i])])
            self.graph_executor.summary_writer.add_summary(summary, timestep)
            episode_reward += rewards[i]
            if terminals[i]:
                summary = tf.Summary(value=[
                    tf.Summary.Value(tag="reward/episode_reward", simple_value=episode_reward)])
                self.graph_executor.summary_writer.add_summary(summary, timestep)
                episode_reward = 0
        timestep += 1
        self.graph_executor.execute(
            ("set_episode_reward", [episode_reward]),
            ("update_global_timestep", [len(rewards)])
        )

    def update(self, batch=None, time_percentage=None, **kwargs):
        """
        Performs an update on the computation graph either via externally experience or
        by sampling from an internal memory.

        Args:
            batch (Optional[dict]): Optional external data batch to use for update. If None, the
                agent should be configured to sample internally.

            time_percentage (Optional[float]): A percentage value (between 0.0 and 1.0) of the time already passed until
                some max timesteps have been reached. This can be used by the algorithm to decay certain parameters
                (e.g. learning rate) over time.

        Returns:
            Union(list, tuple, float): The loss value calculated in this update.
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

            add_timestep (bool): Indicates if current training step should be appended to exported model.
                If false, may override previous checkpoints.
        """
        self.graph_executor.store_model(path=path, add_timestep=add_timestep)

    def load_model(self, checkpoint_directory=None, checkpoint_path=None):
        """
        Loads model from specified path location using the following semantics:

        If checkpoint directory and checkpoint path are given, attempts to find `checkpoint_path` as relative path from
        `checkpoint_directory`.

        If a checkpoint directory is given but no path (e.g. because timestep of checkpoint is not known in advance),
        attempts to fetch latest check-point.

        If no directory is given, attempts to fetch checkpoint from the full absolute path `checkpoint_path'.

        Args:
            checkpoint_directory (str): Optional path to directory containing checkpoint(s).
            checkpoint_path (str): Path to specific model checkpoint.
        """
        self.graph_executor.load_model(checkpoint_directory=checkpoint_directory, checkpoint_path=checkpoint_path)

    def get_weights(self):
        """
        Returns all weights relevant for the agent's policy for syncing purposes.

        Returns:
            any: Weights and optionally weight meta data for this model.
        """
        return self.graph_executor.execute("get_weights")

    def set_weights(self, policy_weights, value_function_weights=None):
        """
        Sets policy weights of this agent, e.g. for external syncing purposes.

        Args:
            policy_weights (any): Weights and optionally meta data to update depending on the backend.
            value_function_weights (Optional[any]): Optional value function weights.

        Raises:
            ValueError if weights do not match graph weights in shapes and types.
        """
        # TODO generic *args here and specific names in specific agents?
        if value_function_weights is not None:
            return self.graph_executor.execute(("set_weights", [policy_weights, value_function_weights]))
        else:
            return self.graph_executor.execute(("set_weights", policy_weights))

    def post_process(self, batch):
        """
        Optional method to post-processes a batch if post-processing is off-loaded to workers instead of
        executed by a central learner before computing the loss.

        The post-processing function must be able to post-process batches of multiple environments
        and episodes with non-terminated fragments via sequence-indices.

        This enables efficient processing of multi-environment batches.

        Args:
            batch (dict): Batch to process. Must contain key 'sequence-indices' to describe where
                environment fragments end (even if the corresponding episode has not terminated.

        Returns:
            any: Post-processed batch.
        """
        pass

    def __repr__(self):
        """
        Returns:
            str: A short, but informative description for this Agent.
        """
        raise NotImplementedError


class IMPALAAgent(OldBaseAgent):
    """
    An Agent implementing the IMPALA algorithm described in [1]. The Agent contains both learner and actor
    API-methods, which will be put into the graph depending on the type ().

    [1] IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures - Espeholt, Soyer,
        Munos et al. - 2018 (https://arxiv.org/abs/1802.01561)
    """

    default_internal_states_space = Tuple(FloatBox(shape=(256,)), FloatBox(shape=(256,)), add_batch_rank=False)
    default_environment_spec = dict(
        type="deepmind_lab", level_id="seekavoid_arena_01", observations=["RGB_INTERLEAVED", "INSTR"],
        frameskip=4
    )

    def __init__(self, discount=0.99, fifo_queue_spec=None, architecture="large", environment_spec=None,
                 feed_previous_action_through_nn=True, feed_previous_reward_through_nn=True,
                 weight_pg=None, weight_vf=None, weight_entropy=None, worker_sample_size=100,
                 **kwargs):
        """
        Args:
            discount (float): The discount factor gamma.
            architecture (str): Which IMPALA architecture to use. One of "small" or "large". Will be ignored if
                `network_spec` is given explicitly in kwargs. Default: "large".
            fifo_queue_spec (Optional[dict,FIFOQueue]): The spec for the FIFOQueue to use for the IMPALA algorithm.
            environment_spec (dict): The spec for constructing an Environment object for an actor-type IMPALA agent.
            feed_previous_action_through_nn (bool): Whether to add the previous action as another input channel to the
                ActionComponent's (NN's) input at each step. This is only possible if the state space is already a Dict.
                It will be added under the key "previous_action". Default: True.
            feed_previous_reward_through_nn (bool): Whether to add the previous reward as another input channel to the
                ActionComponent's (NN's) input at each step. This is only possible if the state space is already a Dict.
                It will be added under the key "previous_reward". Default: True.
            weight_pg (float): See IMPALALossFunction Component.
            weight_vf (float): See IMPALALossFunction Component.
            weight_entropy (float): See IMPALALossFunction Component.
            worker_sample_size (int): How many steps the actor will perform in the environment each sample-run.

        Keyword Args:
            type (str): One of "single", "actor" or "learner". Default: "single".
        """
        type_ = kwargs.pop("type", "single")
        assert type_ in ["single", "actor", "learner"]
        self.type = type_
        self.worker_sample_size = worker_sample_size

        # Network-spec by default is a "large architecture" IMPALA network.
        self.network_spec = kwargs.pop(
            "network_spec",
            dict(type="rlgraph.components.neural_networks.impala.impala_networks.{}IMPALANetwork".
                 format("Large" if architecture == "large" else "Small"))
        )
        if isinstance(self.network_spec, dict) and "type" in self.network_spec and \
                "IMPALANetwork" in self.network_spec["type"]:
            self.network_spec = default_dict(
                self.network_spec,
                dict(worker_sample_size=1 if self.type == "actor" else self.worker_sample_size + 1)
            )

        # Depending on the job-type, remove the pieces from the Agent-spec/graph we won't need.
        self.exploration_spec = kwargs.pop("exploration_spec", None)
        optimizer_spec = kwargs.pop("optimizer_spec", None)
        #observe_spec = kwargs.pop("observe_spec", None)

        self.feed_previous_action_through_nn = feed_previous_action_through_nn
        self.feed_previous_reward_through_nn = feed_previous_reward_through_nn

        # Run everything in a single process.
        if self.type == "single":
            environment_spec = environment_spec or self.default_environment_spec
            #update_spec = kwargs.pop("update_spec", None)
        # Actors won't need to learn (no optimizer needed in graph).
        elif self.type == "actor":
            optimizer_spec = None
            #update_spec = kwargs.pop("update_spec", dict(do_updates=False))
            environment_spec = environment_spec or self.default_environment_spec
        # Learners won't need to explore (act) or observe (insert into Queue).
        else:
            observe_spec = None
            #update_spec = kwargs.pop("update_spec", None)
            environment_spec = None

        # Add previous-action/reward preprocessors to env-specific preprocessor spec.
        self.preprocessing_spec = kwargs.pop("preprocessing_spec", None)
        if self.preprocessing_spec is None:
            config = dict()
            # Flatten actions.
            if self.feed_previous_action_through_nn is True:
                config["previous_action"] = [
                    dict(type="reshape", flatten=True, flatten_categories=kwargs.get("action_space").num_categories)
                ]
            # Bump reward and convert to float32, so that it can be concatenated by the Concat layer.
            if self.feed_previous_reward_through_nn is True:
                config["previous_reward"] = [dict(type="reshape", new_shape=(1,))]

            if len(config) > 0:
                self.preprocessing_spec = dict(type="dict-preprocessor-stack", preprocessors=config)

        # Limit communication in distributed mode between each actor and the learner (never between actors).
        execution_spec = kwargs.pop("execution_spec", None)
        if execution_spec is not None and execution_spec.get("mode") == "distributed":
            default_dict(execution_spec["session_config"], dict(
                type="monitored-training-session",
                allow_soft_placement=True,
                device_filters=["/job:learner/task:0"] + (
                    ["/job:actor/task:{}".format(execution_spec["distributed_spec"]["task_index"])] if
                    self.type == "actor" else ["/job:learner/task:0"]
                )
            ))
            # If Actor, make non-chief in either case (even if task idx == 0).
            if self.type == "actor":
                execution_spec["distributed_spec"]["is_chief"] = False
                # Hard-set device to the CPU for actors.
                execution_spec["device_strategy"] = "custom"
                execution_spec["default_device"] = "/job:{}/task:{}/cpu".format(self.type, execution_spec["distributed_spec"]["task_index"])

        self.policy_spec = kwargs.pop("policy_spec", dict())
        # TODO: Create some auto-setting based on LSTM inside the NN.
        default_dict(self.policy_spec, dict(
            type="shared-value-function-policy",
            deterministic=False,
            reuse_variable_scope="shared-policy",
            action_space=kwargs.get("action_space")
        ))

        # Now that we fixed the Agent's spec, call the super constructor.
        super(IMPALAAgent, self).__init__(
            discount=discount,
            preprocessing_spec=self.preprocessing_spec,
            network_spec=self.network_spec,
            policy_spec=self.policy_spec,
            exploration_spec=self.exploration_spec,
            optimizer_spec=optimizer_spec,
            #observe_spec=observe_spec,
            #update_spec=update_spec,
            execution_spec=execution_spec,
            name=kwargs.pop("name", "impala-{}-agent".format(self.type)),
            **kwargs
        )
        # Always use 1st learner as the parameter server for all policy variables.
        if self.execution_spec["mode"] == "distributed" and self.execution_spec["distributed_spec"]["cluster_spec"]:
            self.policy.propagate_sub_component_properties(dict(device=dict(variables="/job:learner/task:0/cpu")))

        # Check whether we have an RNN.
        self.has_rnn = self.policy.neural_network.has_rnn()
        # Check, whether we are running with GPU.
        self.has_gpu = self.execution_spec["gpu_spec"]["gpus_enabled"] is True and \
            self.execution_spec["gpu_spec"]["num_gpus"] > 0

        # Some FIFO-queue specs.
        self.fifo_queue_keys = ["terminals", "states"] + \
                               (["actions"] if not self.feed_previous_action_through_nn else []) + \
                               (["rewards"] if not self.feed_previous_reward_through_nn else []) + \
                               ["action_probs"] + \
                               (["initial_internal_states"] if self.has_rnn else [])
        # Define FIFO record space.
        # Note that only states and internal_states (RNN) contain num-steps+1 items, all other sub-records only contain
        # num-steps items.
        self.fifo_record_space = Dict(
            {
                "terminals": bool,
                "action_probs": FloatBox(shape=(self.action_space.num_categories,)),
            }, add_batch_rank=False, add_time_rank=self.worker_sample_size
        )
        self.fifo_record_space["states"] = self.state_space.with_time_rank(self.worker_sample_size + 1)
        # Add action and rewards to state or do they have an extra channel?
        if self.feed_previous_action_through_nn:
            self.fifo_record_space["states"]["previous_action"] = \
                self.action_space.with_time_rank(self.worker_sample_size + 1)
        else:
            self.fifo_record_space["actions"] = self.action_space.with_time_rank(self.worker_sample_size)
        if self.feed_previous_action_through_nn:
            self.fifo_record_space["states"]["previous_reward"] = FloatBox(add_time_rank=self.worker_sample_size + 1)
        else:
            self.fifo_record_space["rewards"] = FloatBox(add_time_rank=self.worker_sample_size)

        if self.has_rnn:
            self.fifo_record_space["initial_internal_states"] = self.internal_states_space.with_time_rank(False)

        # Create our FIFOQueue (actors will enqueue, learner(s) will dequeue).
        self.fifo_queue = FIFOQueue.from_spec(
            fifo_queue_spec or dict(capacity=1),
            reuse_variable_scope="shared-fifo-queue",
            only_insert_single_records=True,
            record_space=self.fifo_record_space,
            device="/job:learner/task:0/cpu" if self.execution_spec["mode"] == "distributed" and
            self.execution_spec["distributed_spec"]["cluster_spec"] else None
        )

        # Remove `states` key from input_spaces: not needed.
        del self.input_spaces["states"]

        # Add all our sub-components to the core.
        if self.type == "single":
            pass

        elif self.type == "actor":
            # No learning, no loss function.
            self.loss_function = None
            # A Dict Splitter to split things from the EnvStepper.
            self.env_output_splitter = ContainerSplitter(tuple_length=4, scope="env-output-splitter")

            self.states_dict_splitter = None

            # Slice some data from the EnvStepper (e.g only first internal states are needed).
            self.internal_states_slicer = Slice(scope="internal-states-slicer", squeeze=True)
            # Merge back to insert into FIFO.
            self.fifo_input_merger = ContainerMerger(*self.fifo_queue_keys)

            # Dummy Flattener to calculate action-probs space.
            dummy_flattener = ReShape(flatten=True, flatten_categories=self.action_space.num_categories)
            self.environment_stepper = EnvironmentStepper(
                environment_spec=environment_spec,
                actor_component_spec=ActorComponent(self.preprocessor, self.policy, self.exploration),
                state_space=self.state_space.with_batch_rank(),
                reward_space=float,  # TODO <- float64 for deepmind? may not work for other envs
                internal_states_space=self.internal_states_space,
                num_steps=self.worker_sample_size,
                add_previous_action_to_state=True,
                add_previous_reward_to_state=True,
                add_action_probs=True,
                action_probs_space=dummy_flattener.get_preprocessed_space(self.action_space)
            )
            sub_components = [
                self.environment_stepper, self.env_output_splitter,
                self.internal_states_slicer, self.fifo_input_merger,
                self.fifo_queue
            ]
        # Learner.
        else:
            self.environment_stepper = None

            # A Dict splitter to split up items from the queue.
            self.fifo_input_merger = None
            self.fifo_output_splitter = ContainerSplitter(*self.fifo_queue_keys, scope="fifo-output-splitter")
            self.states_dict_splitter = ContainerSplitter(
                *list(self.fifo_record_space["states"].keys()), scope="states-dict-splitter"
            )
            self.internal_states_slicer = None

            self.transposer = Transpose(
                scope="transposer",
                device=dict(ops="/job:learner/task:0/cpu")
            )
            self.staging_area = StagingArea(num_data=len(self.fifo_queue_keys))

            # Create an IMPALALossFunction with some parameters.
            self.loss_function = IMPALALossFunction(
                discount=self.discount, weight_pg=weight_pg, weight_vf=weight_vf,
                weight_entropy=weight_entropy,
                slice_actions=self.feed_previous_action_through_nn,
                slice_rewards=self.feed_previous_reward_through_nn,
                device="/job:learner/task:0/gpu"
            )

            self.policy.propagate_sub_component_properties(
                dict(device=dict(variables="/job:learner/task:0/cpu", ops="/job:learner/task:0/gpu"))
            )
            for component in [self.staging_area, self.preprocessor, self.optimizer]:
                component.propagate_sub_component_properties(
                    dict(device="/job:learner/task:0/gpu")
                )

            sub_components = [
                self.fifo_output_splitter, self.fifo_queue, self.states_dict_splitter,
                self.transposer,
                self.staging_area, self.preprocessor, self.policy,
                self.loss_function, self.optimizer
            ]

        if self.type != "single":
            # Add all the agent's sub-components to the root.
            self.root_component.add_components(*sub_components)

            # Define the Agent's (root Component's) API.
            self.define_graph_api(*sub_components)

        if self.type != "single" and self.auto_build:
            if self.type == "learner":
                build_options = dict(
                    build_device_context="/job:learner/task:0/cpu",
                    pin_global_variable_device="/job:learner/task:0/cpu"
                )
                self._build_graph([self.root_component], self.input_spaces, optimizer=self.optimizer,
                                  build_options=build_options)
            else:
                self._build_graph([self.root_component], self.input_spaces, optimizer=self.optimizer,
                                  build_options=None)

            self.graph_built = True

            if self.has_gpu:
                # Get 1st return op of API-method `stage` of sub-component `staging-area` (which is the stage-op).
                self.stage_op = self.root_component.sub_components["staging-area"].api_methods["stage"]. \
                    out_op_columns[0].op_records[0].op
                # Initialize the stage.
                self.graph_executor.monitored_session.run_step_fn(
                    lambda step_context: step_context.session.run(self.stage_op)
                )

                # TODO remove after full refactor.
                self.dequeue_op = self.root_component.sub_components["fifo-queue"].api_methods["get_records"]. \
                    out_op_columns[0].op_records[0].op
            if self.type == "actor":
                self.enqueue_op = self.root_component.sub_components["fifo-queue"].api_methods["insert_records"]. \
                    out_op_columns[0].op_records[0].op

    def define_graph_api(self, *sub_components):
        # TODO: Unify agents with/w/o synchronizable policy.
        # TODO: Unify Agents with/w/o get_action method (w/ env-stepper vs w/o).
        #global_scope_base = "environment-stepper/actor-component/" if self.type == "actor" else ""
        #super(IMPALAAgent, self).define_graph_api(
        #    global_scope_base+"policy",
        #    global_scope_base+"dict-preprocessor-stack"
        #)

        # Assemble the specific agent.
        if self.type == "single":
            pass
        elif self.type == "actor":
            self.define_graph_api_actor(*sub_components)
        else:
            self.define_graph_api_learner(*sub_components)

    def define_graph_api_actor(self, env_stepper, env_output_splitter, internal_states_slicer, merger, fifo_queue):
        """
        Defines the API-methods used by an IMPALA actor. Actors only step through an environment (n-steps at
        a time), collect the results and push them into the FIFO queue. Results include: The actions actually
        taken, the discounted accumulated returns for each action, the probability of each taken action according to
        the behavior policy.

        Args:
            env_stepper (EnvironmentStepper): The EnvironmentStepper Component to setp through the Env n steps
                in a single op call.

            fifo_queue (FIFOQueue): The FIFOQueue Component used to enqueue env sample runs (n-step).
        """
        # Perform n-steps in the env and insert the results into our FIFO-queue.
        @rlgraph_api(component=self.root_component)
        def perform_n_steps_and_insert_into_fifo(root):
            # Take n steps in the environment.
            step_results = env_stepper.step()

            split_output = env_output_splitter.call(step_results)
            # Slice off the initial internal state (so the learner can re-feed-forward from that internal-state).
            initial_internal_states = internal_states_slicer.slice(split_output[-1], 0)  # -1=internal states
            to_merge = split_output[:-1] + (initial_internal_states,)
            record = merger.merge(*to_merge)

            # Insert results into the FIFOQueue.
            insert_op = fifo_queue.insert_records(record)

            return insert_op, split_output[0]  # 0=terminals

    def define_graph_api_learner(
            self, fifo_output_splitter, fifo_queue, states_dict_splitter,
            transposer, staging_area, preprocessor, policy, loss_function, optimizer
    ):
        """
        Defines the API-methods used by an IMPALA learner. Its job is basically: Pull a batch from the
        FIFOQueue, split it up into its components and pass these through the loss function and into the optimizer for
        a learning update.

        Args:
            fifo_output_splitter (ContainerSplitter): The ContainerSplitter Component to split up a batch from the queue
                along its items.

            fifo_queue (FIFOQueue): The FIFOQueue Component used to enqueue env sample runs (n-step).

            states_dict_splitter (ContainerSplitter): The ContainerSplitter Component to split the state components
                into its single parts.

            transposer (Transpose): A space-agnostic Transpose to flip batch- and time ranks of all state-components.
            staging_area (StagingArea): A possible GPU stating area component.

            preprocessor (PreprocessorStack): A preprocessing Component for the states (may be a DictPreprocessorStack
                as well).

            policy (Policy): The Policy Component, which to update.
            loss_function (IMPALALossFunction): The IMPALALossFunction Component.
            optimizer (Optimizer): The optimizer that we use to calculate an update and apply it.
        """
        @rlgraph_api(component=self.root_component)
        def get_queue_size(root):
            return fifo_queue.get_size()

        @rlgraph_api(component=self.root_component)
        def update_from_memory(root, time_percentage=None):
            # Pull n records from the queue.
            # Note that everything will come out as batch-major and must be transposed before the main-LSTM.
            # This is done by the network itself for all network inputs:
            # - preprocessed_s
            # - preprocessed_last_s_prime
            # But must still be done for actions, rewards, terminals here in this API-method via separate ReShapers.
            records = fifo_queue.get_records(self.update_spec["batch_size"])

            split_record = fifo_output_splitter.call(records)
            actions = None
            rewards = None
            if self.feed_previous_action_through_nn and self.feed_previous_reward_through_nn:
                terminals, states, action_probs_mu, initial_internal_states = split_record
            else:
                terminals, states, actions, rewards, action_probs_mu, initial_internal_states = split_record

            # Flip everything to time-major.
            # TODO: Create components that are less input-space sensitive (those that have no variables should
            # TODO: be reused for any kind of processing)
            states = transposer.call(states)
            terminals = transposer.call(terminals)
            action_probs_mu = transposer.call(action_probs_mu)
            if self.feed_previous_action_through_nn is False:
                actions = transposer.call(actions)
            if self.feed_previous_reward_through_nn is False:
                rewards = transposer.call(rewards)

            # If we use a GPU: Put everything on staging area (adds 1 time step policy lag, but makes copying
            # data into GPU more efficient).
            if self.has_gpu:
                stage_op = staging_area.stage(states, terminals, action_probs_mu, initial_internal_states)
                # Get data from stage again and continue.
                states, terminals, action_probs_mu, initial_internal_states = staging_area.unstage()
            else:
                # TODO: No-op component?
                stage_op = None

            # Preprocess actions and rewards inside the state (actions: flatten one-hot, rewards: expand).
            preprocessed_states = preprocessor.preprocess(states)

            # Only retrieve logits and do faster sparse softmax in loss.
            out = policy.get_state_values_adapter_outputs_and_parameters(preprocessed_states, initial_internal_states)
            state_values_pi = out["state_values"]
            logits = out["adapter_outputs"]
            #current_internal_states = out["last_internal_states"]

            # Isolate actions and rewards from states.
            if self.feed_previous_action_through_nn or self.feed_previous_reward_through_nn:
                states_split = states_dict_splitter.call(states)
                actions = states_split[-2]
                rewards = states_split[-1]

            # Calculate the loss.
            loss, loss_per_item = loss_function.loss(
                logits, action_probs_mu, state_values_pi, actions, rewards, terminals
            )
            policy_vars = policy.variables()

            # Pass vars and loss values into optimizer.
            step_op = optimizer.step(policy_vars, loss, loss_per_item, time_percentage)
            # Increase the global training step counter.
            step_op = root._graph_fn_training_step(step_op)

            # Return optimizer op and all loss values.
            # TODO: Make it possible to return None from API-method without messing with the meta-graph.
            return step_op, (stage_op if stage_op else step_op), loss, loss_per_item, records

    # TODO: Unify with other Agents.
    def get_action(self, states, internals=None, use_exploration=True, apply_preprocessing=True, extra_returns=None,
                   time_percentage=None):
        pass

    def _observe_graph(self, preprocessed_states, actions, internals, rewards, terminals, **kwargs):
        self.graph_executor.execute(("insert_records", [preprocessed_states, actions, rewards, terminals]))

    def update(self, batch=None, time_percentage=None, sequence_indices=None, apply_postprocessing=True):
        q_size = self.graph_executor.execute("get_queue_size")
        self.timesteps += q_size * self.worker_sample_size

        if time_percentage is None:
            time_percentage = self.timesteps / (self.max_timesteps or 1e6)

        self.num_updates += 1

        if batch is None:
            # Include stage_op or not?
            if self.has_gpu:
                return self.graph_executor.execute(("update_from_memory", time_percentage))
            else:
                return self.graph_executor.execute(("update_from_memory", time_percentage, ([0, 2, 3, 4])))
        else:
            raise RLGraphError("Cannot call update-from-batch on an IMPALA Agent.")

    def __repr__(self):
        return "IMPALAAgent(type={})".format(self.type)


class SingleIMPALAAgent(IMPALAAgent):
    """
    A single IMPALAAgent, performing both experience collection and learning updates via multi-threading
    (queue runners).
    """
    def __init__(self, discount=0.99, fifo_queue_spec=None, architecture="large", environment_spec=None,
                 feed_previous_action_through_nn=True, feed_previous_reward_through_nn=True,
                 weight_pg=None, weight_vf=None, weight_entropy=None,
                 num_workers=1, worker_sample_size=100,
                 dynamic_batching=False, visualize=False, **kwargs):
        """
        Args:
            discount (float): The discount factor gamma.
            architecture (str): Which IMPALA architecture to use. One of "small" or "large". Will be ignored if
                `network_spec` is given explicitly in kwargs. Default: "large".
            fifo_queue_spec (Optional[dict,FIFOQueue]): The spec for the FIFOQueue to use for the IMPALA algorithm.
            environment_spec (dict): The spec for constructing an Environment object for an actor-type IMPALA agent.
            feed_previous_action_through_nn (bool): Whether to add the previous action as another input channel to the
                ActionComponent's (NN's) input at each step. This is only possible if the state space is already a Dict.
                It will be added under the key "previous_action". Default: True.
            feed_previous_reward_through_nn (bool): Whether to add the previous reward as another input channel to the
                ActionComponent's (NN's) input at each step. This is only possible if the state space is already a Dict.
                It will be added under the key "previous_reward". Default: True.
            weight_pg (float): See IMPALALossFunction Component.
            weight_vf (float): See IMPALALossFunction Component.
            weight_entropy (float): See IMPALALossFunction Component.
            num_workers (int): How many actors (workers) should be run in separate threads.
            worker_sample_size (int): How many steps the actor will perform in the environment each sample-run.
            dynamic_batching (bool): Whether to use the deepmind's custom dynamic batching op for wrapping the
                optimizer's step call. The batcher.so file must be compiled for this to work (see Docker file).
                Default: False.
            visualize (Union[int,bool]): Whether and how many workers to visualize.
                Default: False (no visualization).
        """
        # Now that we fixed the Agent's spec, call the super constructor.
        super(SingleIMPALAAgent, self).__init__(
            type="single",
            discount=discount,
            architecture=architecture,
            fifo_queue_spec=fifo_queue_spec,
            environment_spec=environment_spec,
            feed_previous_action_through_nn=feed_previous_action_through_nn,
            feed_previous_reward_through_nn=feed_previous_reward_through_nn,
            weight_pg=weight_pg,
            weight_vf=weight_vf,
            weight_entropy=weight_entropy,
            worker_sample_size=worker_sample_size,
            name=kwargs.pop("name", "impala-single-agent"),
            **kwargs
        )
        self.dynamic_batching = dynamic_batching
        self.num_workers = num_workers
        self.visualize = visualize

        # If we use dynamic batching, wrap the dynamic batcher around the policy's graph_fn that we
        # actually call below during our build.
        if self.dynamic_batching:
            self.policy = DynamicBatchingPolicy(policy_spec=self.policy, scope="")

        self.env_output_splitter = ContainerSplitter(
            tuple_length=3 if self.has_rnn is False else 4, scope="env-output-splitter"
        )
        self.fifo_output_splitter = ContainerSplitter(*self.fifo_queue_keys, scope="fifo-output-splitter")
        self.states_dict_splitter = ContainerSplitter(
            *list(self.fifo_record_space["states"].keys() if isinstance(self.state_space, Dict) else "dummy"),
            scope="states-dict-splitter"
        )

        self.staging_area = StagingArea(num_data=len(self.fifo_queue_keys))

        # Slice some data from the EnvStepper (e.g only first internal states are needed).
        if self.has_rnn:
            internal_states_slicer = Slice(scope="internal-states-slicer", squeeze=True)
        else:
            internal_states_slicer = None

        self.transposer = Transpose(scope="transposer")

        # Create an IMPALALossFunction with some parameters.
        self.loss_function = IMPALALossFunction(
            discount=self.discount, weight_pg=weight_pg, weight_vf=weight_vf,
            weight_entropy=weight_entropy, slice_actions=self.feed_previous_action_through_nn,
            slice_rewards=self.feed_previous_reward_through_nn
        )

        # Merge back to insert into FIFO.
        self.fifo_input_merger = ContainerMerger(*self.fifo_queue_keys)

        # Dummy Flattener to calculate action-probs space.
        dummy_flattener = ReShape(flatten=True, flatten_categories=self.action_space.num_categories)

        self.environment_steppers = list()
        for i in range(self.num_workers):
            environment_spec_ = copy.deepcopy(environment_spec)
            if self.visualize is True or (isinstance(self.visualize, int) and i+1 <= self.visualize):
                environment_spec_["visualize"] = True

            # Force worker_sample_size for IMPALA NNs (LSTM) in env-stepper to be 1.
            policy_spec = copy.deepcopy(self.policy_spec)
            if isinstance(policy_spec, dict) and isinstance(policy_spec["network_spec"], dict) and \
                    "type" in policy_spec["network_spec"] and "IMPALANetwork" in policy_spec["network_spec"]["type"]:
                policy_spec["network_spec"]["worker_sample_size"] = 1

            env_stepper = EnvironmentStepper(
                environment_spec=environment_spec_,
                actor_component_spec=ActorComponent(
                    preprocessor_spec=self.preprocessing_spec,
                    policy_spec=policy_spec,
                    exploration_spec=self.exploration_spec
                ),
                state_space=self.state_space.with_batch_rank(),
                action_space=self.action_space.with_batch_rank(),
                reward_space=float,
                internal_states_space=self.internal_states_space,
                num_steps=self.worker_sample_size,
                add_action=not self.feed_previous_action_through_nn,
                add_reward=not self.feed_previous_reward_through_nn,
                add_previous_action_to_state=self.feed_previous_action_through_nn,
                add_previous_reward_to_state=self.feed_previous_reward_through_nn,
                add_action_probs=True,
                action_probs_space=dummy_flattener.get_preprocessed_space(self.action_space),
                scope="env-stepper-{}".format(i)
            )
            if self.dynamic_batching:
                env_stepper.actor_component.policy.parent_component = None
                env_stepper.actor_component.policy = DynamicBatchingPolicy(
                    policy_spec=env_stepper.actor_component.policy, scope="")
                env_stepper.actor_component.add_components(env_stepper.actor_component.policy)

            self.environment_steppers.append(env_stepper)

        # Create the QueueRunners (one for each env-stepper).
        self.queue_runner = QueueRunner(
            self.fifo_queue, "step", -1,  # -1: Take entire return value of API-method `step` as record to insert.
            self.env_output_splitter,
            self.fifo_input_merger,
            internal_states_slicer,
            *self.environment_steppers
        )

        sub_components = [
            self.fifo_output_splitter, self.fifo_queue, self.queue_runner,
            self.transposer,
            self.staging_area, self.preprocessor, self.states_dict_splitter,
            self.policy, self.loss_function, self.optimizer
        ]

        # Add all the agent's sub-components to the root.
        self.root_component.add_components(*sub_components)

        # Define the Agent's (root Component's) API.
        self.define_graph_api()

        if self.auto_build:
            self._build_graph([self.root_component], self.input_spaces)
            self.graph_built = True

            if self.has_gpu:
                # Get 1st return op of API-method `stage` of sub-component `staging-area` (which is the stage-op).
                self.stage_op = self.root_component.sub_components["staging-area"].api_methods["stage"]. \
                    out_op_columns[0].op_records[0].op
                # Initialize the stage.
                self.graph_executor.monitored_session.run_step_fn(
                    lambda step_context: step_context.session.run(self.stage_op)
                )
                # TODO remove after full refactor.
                self.dequeue_op = self.root_component.sub_components["fifo-queue"].api_methods["get_records"]. \
                    out_op_columns[0].op_records[0].op

    def define_graph_api(self):
        agent = self

        @rlgraph_api(component=self.root_component)
        def setup_queue_runner(root):
            return agent.queue_runner.setup()

        @rlgraph_api(component=self.root_component)
        def get_queue_size(root):
            return agent.fifo_queue.get_size()

        @rlgraph_api(component=self.root_component)
        def update_from_memory(root, time_percentage=None):
            # Pull n records from the queue.
            # Note that everything will come out as batch-major and must be transposed before the main-LSTM.
            # This is done by the network itself for all network inputs:
            # - preprocessed_s
            # - preprocessed_last_s_prime
            # But must still be done for actions, rewards, terminals here in this API-method via separate ReShapers.
            records = agent.fifo_queue.get_records(self.update_spec["batch_size"])

            out = agent.fifo_output_splitter.split_into_dict(records)
            terminals = out["terminals"]
            states = out["states"]
            action_probs_mu = out["action_probs"]
            initial_internal_states = None
            if self.has_rnn:
                initial_internal_states = out["initial_internal_states"]

            # Flip everything to time-major.
            # TODO: Create components that are less input-space sensitive (those that have no variables should
            # TODO: be reused for any kind of processing: already done, use space_agnostic feature. See ReShape)
            states = agent.transposer.call(states)
            terminals = agent.transposer.call(terminals)
            action_probs_mu = agent.transposer.call(action_probs_mu)
            actions = None
            if not self.feed_previous_action_through_nn:
                actions = agent.transposer.call(out["actions"])
            rewards = None
            if not self.feed_previous_reward_through_nn:
                rewards = agent.transposer.call(out["rewards"])

            # If we use a GPU: Put everything on staging area (adds 1 time step policy lag, but makes copying
            # data into GPU more efficient).
            if self.has_gpu:
                if self.has_rnn:
                    stage_op = agent.staging_area.stage(states, terminals, action_probs_mu, initial_internal_states)
                    states, terminals, action_probs_mu, initial_internal_states = agent.staging_area.unstage()
                else:
                    stage_op = agent.staging_area.stage(states, terminals, action_probs_mu)
                    states, terminals, action_probs_mu = agent.staging_area.unstage()
            else:
                # TODO: No-op component?
                stage_op = None

            # Preprocess actions and rewards inside the state (actions: flatten one-hot, rewards: expand).
            if agent.preprocessing_required:
                states = agent.preprocessor.preprocess(states)

            # Get the pi-action probs AND the values for all our states.
            out = agent.policy.get_state_values_adapter_outputs_and_parameters(states, initial_internal_states)
            state_values_pi = out["state_values"]
            logits_pi = out["adapter_outputs"]

            # Isolate actions and rewards from states.
            # TODO: What if only one of actions or rewards is fed through NN, but the other not?
            if self.feed_previous_reward_through_nn and self.feed_previous_action_through_nn:
                out = agent.states_dict_splitter.call(states)
                actions = out[-2]  # TODO: Are these always the correct slots for "previous_action" and "previous_reward"?
                rewards = out[-1]

            # Calculate the loss.
            loss, loss_per_item = agent.loss_function.loss(
                logits_pi, action_probs_mu, state_values_pi, actions, rewards, terminals
            )
            if self.dynamic_batching:
                policy_vars = agent.queue_runner.data_producing_components[0].actor_component.policy.variables()
            else:
                policy_vars = agent.policy.variables()

            # Pass vars and loss values into optimizer.
            step_op = agent.optimizer.step(policy_vars, loss, loss_per_item, time_percentage)
            # Increase the global training step counter.
            step_op = root._graph_fn_training_step(step_op)

            # Return optimizer op and all loss values.
            # TODO: Make it possible to return None from API-method without messing with the meta-graph.
            return step_op, (stage_op if stage_op else step_op), loss, loss_per_item, records

        # TODO: Move this into generic AgentRootComponent.
        @graph_fn(component=self.root_component)
        def _graph_fn_training_step(root, other_step_op=None):
            add_op = tf.assign_add(self.graph_executor.global_training_timestep, 1)
            op_list = [add_op] + [other_step_op] if other_step_op is not None else []
            with tf.control_dependencies(op_list):
                return tf.no_op() if other_step_op is None else other_step_op

    def __repr__(self):
        return "SingleIMPALAAgent()"
