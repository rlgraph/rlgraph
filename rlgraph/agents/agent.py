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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from collections import defaultdict
from functools import partial

import numpy as np

from rlgraph import get_backend
from rlgraph.components import Component, Exploration, PreprocessorStack, Synchronizable, Policy, Optimizer, \
    ContainerMerger, ContainerSplitter
from rlgraph.graphs.graph_builder import GraphBuilder
from rlgraph.graphs.graph_executor import GraphExecutor
from rlgraph.spaces import Space, ContainerSpace
from rlgraph.utils.decorators import rlgraph_api, graph_fn
from rlgraph.utils.input_parsing import parse_execution_spec, parse_observe_spec, parse_update_spec, \
    parse_value_function_spec
from rlgraph.utils.ops import flatten_op
from rlgraph.utils.specifiable import Specifiable

if get_backend() == "tf":
    import tensorflow as tf


class Agent(Specifiable):
    """
    Generic agent defining RLGraph-API operations and parses and sanitizes configuration specs.
    """

    def __init__(self, state_space, action_space, discount=0.98,
                 preprocessing_spec=None, network_spec=None, internal_states_space=None,
                 policy_spec=None, value_function_spec=None,
                 exploration_spec=None, execution_spec=None, optimizer_spec=None, value_function_optimizer_spec=None,
                 observe_spec=None, update_spec=None,
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
            time_percentage=float
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
        self.value_function = parse_value_function_spec(value_function_spec)
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
            if i < 2:
                return []
            return tuple([[] for _ in range(i)])

        self.states_buffer = defaultdict(list)  # partial(fact_, len(self.flat_state_space)))
        self.actions_buffer = defaultdict(partial(factory_, len(self.flat_action_space or [])))
        self.internals_buffer = defaultdict(list)
        self.rewards_buffer = defaultdict(list)
        self.next_states_buffer = defaultdict(list)  # partial(fact_, len(self.flat_state_space)))
        self.terminals_buffer = defaultdict(list)

        self.observe_spec = parse_observe_spec(observe_spec)

        # Global time step counter.
        self.timesteps = 0

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
        del self.states_buffer[env_id]  # = ([] for _ in range(len(self.flat_state_space)))
        del self.actions_buffer[env_id]  # = ([] for _ in range(len(self.flat_action_space)))
        del self.internals_buffer[env_id]  # = []
        del self.rewards_buffer[env_id]  # = []
        del self.next_states_buffer[env_id]  # = ([] for _ in range(len(self.flat_state_space)))
        del self.terminals_buffer[env_id]  # = []

    def define_graph_api(self, *args, **kwargs):
        """
        Can be used to specify and then `self.define_api_method` the Agent's CoreComponent's API methods.
        Each agent implements this to build its algorithm logic.
        """
        agent = self

        if self.value_function is not None:
            # This avoids variable-incompleteness for the value-function component in a multi-GPU setup, where the root
            # value-function never performs any forward pass (only used as variable storage).
            @rlgraph_api(component=self.root_component)
            def get_state_values(root, preprocessed_states):
                vf = root.get_sub_component_by_name(agent.value_function.scope)
                return vf.value_output(preprocessed_states)

        # Add API methods for syncing.
        @rlgraph_api(component=self.root_component)
        def get_weights(root):
            policy = root.get_sub_component_by_name(agent.policy.scope)
            policy_weights = policy.variables()
            value_function_weights = None
            if agent.value_function is not None:
                value_func = root.get_sub_component_by_name(agent.value_function.scope)
                value_function_weights = value_func.variables()
            return dict(policy_weights=policy_weights, value_function_weights=value_function_weights)

        @rlgraph_api(component=self.root_component, must_be_complete=False)
        def set_weights(root, policy_weights, value_function_weights=None):
            policy = root.get_sub_component_by_name(agent.policy.scope)
            policy_sync_op = policy.sync(policy_weights)
            if value_function_weights is not None:
                assert agent.value_function is not None
                vf = root.get_sub_component_by_name(agent.value_function.scope)
                vf_sync_op = vf.sync(value_function_weights)
                return root._graph_fn_group(policy_sync_op, vf_sync_op)
            else:
                return policy_sync_op

        # TODO: Replace this with future on-the-fly-API-components.
        @graph_fn(component=self.root_component)
        def _graph_fn_group(root, *ops):
            if get_backend() == "tf":
                return tf.group(*ops)
            return ops[0]

        # To pre-process external data if needed.
        @rlgraph_api(component=self.root_component)
        def preprocess_states(root, states):
            preprocessor_stack = root.get_sub_component_by_name(agent.preprocessor.scope)
            return preprocessor_stack.preprocess(states)

        @graph_fn(component=self.root_component)
        def _graph_fn_training_step(root, other_step_op=None):
            """
            Increases the global training timestep by 1. Should be called by all training API-methods to
            timestamp each training/update step.

            Args:
                other_step_op (Optional[DataOp]): Another DataOp (e.g. a step_op) which should be
                    executed before the increase takes place.

            Returns:
                DataOp: no_op() or identity(other_step_op) in tf, None in pytorch.
            """
            if get_backend() == "tf":
                add_op = tf.assign_add(self.graph_executor.global_training_timestep, 1)
                op_list = [add_op] + [other_step_op] if other_step_op is not None else []
                with tf.control_dependencies(op_list):
                    if other_step_op is None or hasattr(other_step_op, "type") and other_step_op.type == "NoOp":
                        return tf.no_op()
                    else:
                        return tf.identity(other_step_op)
            elif get_backend == "pytorch":
                self.graph_executor.global_training_timestep += 1
                return None

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
            [self.root_component], self.input_spaces, optimizer=self.optimizer,
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
                self._observe_graph(
                    preprocessed_states=np.asarray(self.states_buffer[env_id]),
                    actions=actions_,
                    internals=np.asarray(self.internals_buffer[env_id]),
                    rewards=np.asarray(self.rewards_buffer[env_id]),
                    next_states=np.asarray(self.next_states_buffer[env_id]),
                    terminals=np.asarray(self.terminals_buffer[env_id])
                )
                self.reset_env_buffers(env_id)
        else:
            if not batched:
                preprocessed_states = self.preprocessed_state_space.force_batch(preprocessed_states)
                next_states = self.preprocessed_state_space.force_batch(next_states)
                actions = self.action_space.force_batch(actions)
                rewards = [rewards]
                terminals = [terminals]

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
