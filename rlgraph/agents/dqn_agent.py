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

from rlgraph.agents import Agent
from rlgraph.components import Memory, PrioritizedReplay, DQNLossFunction
from rlgraph.components.algorithms.algorithm_component import AlgorithmComponent
from rlgraph.components.policies.dueling_policy import DuelingPolicy
from rlgraph.execution.rules.sync_rules import SyncRules
from rlgraph.spaces import FloatBox, BoolBox
from rlgraph.spaces.space_utils import sanity_check_space
from rlgraph.utils.decorators import rlgraph_api
from rlgraph.utils.rlgraph_errors import RLGraphError
from rlgraph.utils.util import strip_list


class DQNAgent(Agent):
    """
    A collection of DQN algorithms published in the following papers:

    [1] Human-level control through deep reinforcement learning. Mnih, Kavukcuoglu, Silver et al. - 2015
    [2] Deep Reinforcement Learning with Double Q-learning. v. Hasselt, Guez, Silver - 2015
    [3] Dueling Network Architectures for Deep Reinforcement Learning, Wang et al. - 2016
    [4] https://en.wikipedia.org/wiki/Huber_loss
    """
    def __init__(
        self,
        state_space,
        action_space,
        *,  # Force all following args as named.
        discount=0.98,
        python_buffer_size=0,
        memory_batch_size=None,
        double_q=True,
        dueling_q=True,
        n_step=1,
        preprocessing_spec=None,
        memory_spec=None,
        internal_states_space=None,
        policy_spec=None,
        network_spec=None,
        exploration_spec=None,
        execution_spec=None,
        optimizer_spec=None,
        observe_spec=None,  # Obsoleted.
        update_spec=None,  # Obsoleted.
        sync_rules=None,
        summary_spec=None,
        saver_spec=None,
        huber_loss=False,
        shared_container_action_target=True,
        auto_build=True,
        name="dqn-agent",
    ):
        """
        Args:
            preprocessing_spec (Optional[list,PreprocessorStack]): The spec list for the different necessary states
                preprocessing steps or a PreprocessorStack object itself.
            discount (float): The discount factor (gamma).
            network_spec (Optional[list,NeuralNetwork]): Spec list for a NeuralNetwork Component or the NeuralNetwork
                object itself.
            internal_states_space (Optional[Union[dict,Space]]): Spec dict for the internal-states Space or a direct
                Space object for the Space(s) of the internal (RNN) states.
            policy_spec (Optional[dict]): An optional dict for further kwargs passing into the Policy c'tor.
            exploration_spec (Optional[dict]): The spec-dict to create the Exploration Component.
            execution_spec (Optional[dict,Execution]): The spec-dict specifying execution settings.
            optimizer_spec (Optional[dict,Optimizer]): The spec-dict to create the Optimizer for this Agent.
            observe_spec (Optional[dict]): Spec-dict to specify `Agent.observe()` settings.
            update_spec (Optional[dict]): Obsoleted: Spec-dict to specify `Agent.update()` settings.
            summary_spec (Optional[dict]): Spec-dict to specify summary settings.
            saver_spec (Optional[dict]): Spec-dict to specify saver settings.
            name (str): Some name for this Agent object.
            double_q (bool): Whether to use the double DQN loss function (see [2]).
            dueling_q (bool): Whether to use a dueling layer in the ActionAdapter  (see [3]).
            huber_loss (bool) : Whether to apply a Huber loss. (see [4]).
            sync_every_n_updates (int): Every how many updates, do we need to sync into the target network?
            n_step (Optional[int]): n-step adjustment to discounting.
            memory_spec (Optional[dict,Memory]): The spec for the Memory to use for the DQN algorithm.
            auto_build (Optional[bool]): If True (default), immediately builds the graph using the agent's
                graph builder. If false, users must separately call agent.build(). Useful for debugging or analyzing
                components before building.
        """
        super(DQNAgent, self).__init__(
            state_space=state_space,
            action_space=action_space,
            python_buffer_size=python_buffer_size,
            #custom_python_buffers=custom_python_buffers,  # why would we need custom buffers for DQN?
            internal_states_space=internal_states_space,
            execution_spec=execution_spec,
            observe_spec=observe_spec,  # Obsoleted.
            update_spec=update_spec,  # Obsoleted.
            summary_spec=summary_spec,
            saver_spec=saver_spec,
            #auto_build=auto_build,
            name=name
        )

        if n_step > 1:
            if self.python_buffer_size == 0:
                raise RLGraphError(
                    "Cannot setup observations with n-step (n={}), while buffering is switched "
                    "off".format(n_step)
                )
            elif self.python_buffer_size < 3 * n_step:
                raise RLGraphError(
                    "Buffer must be at least 3x as large as n-step (3 x n={}, buffer-size={})!".
                    format(n_step, self.python_buffer_size)
                )

        # Keep track of when to sync the target network (every n updates).
        self.sync_rules = SyncRules.from_spec(sync_rules)
        self.steps_since_target_net_sync = 0

        # Change our root-component to PPO.
        self.root_component = DQNAlgorithmComponent(
            agent=self, discount=discount, memory_spec=memory_spec,
            preprocessing_spec=preprocessing_spec, policy_spec=policy_spec, network_spec=network_spec,
            exploration_spec=exploration_spec, optimizer_spec=optimizer_spec,
            memory_batch_size=memory_batch_size, n_step=n_step, double_q=double_q, dueling_q=dueling_q,
            huber_loss=huber_loss, shared_container_action_target=shared_container_action_target
        )

        # Extend input Space definitions to this Agent's specific API-methods.
        self.preprocessed_state_space = self.preprocessed_state_space = self.root_component.preprocessor.\
            get_preprocessed_space(self.state_space).with_batch_rank()

        self.input_spaces.update(dict(
            actions=self.action_space.with_batch_rank(),
            # Weights will have a Space derived from the vars of policy.
            policy_weights="variables:{}".format(self.root_component.policy.scope),
            preprocessed_states=self.preprocessed_state_space,
            rewards=FloatBox(add_batch_rank=True),
            terminals=BoolBox(add_batch_rank=True),
            next_states=self.preprocessed_state_space,
            preprocessed_next_states=self.preprocessed_state_space,
            importance_weights=FloatBox(add_batch_rank=True),
            apply_postprocessing=bool,
            deterministic=bool
        ))

        # Build this Agent's graph.
        if auto_build is True:
            self.build()

    def get_action(self, states, internals=None, use_exploration=True, apply_preprocessing=True, extra_returns=None,
                   time_percentage=None):
        """
        Args:
            extra_returns (Optional[Set[str],str]): Optional string or set of strings for additional return
                values (besides the actions). Possible values are:
                - 'preprocessed_states': The preprocessed states after passing the given states through the
                preprocessor stack.
                - 'internal_states': The internal states returned by the RNNs in the NN pipeline.
                - 'used_exploration': Whether epsilon- or noise-based exploration was used or not.

        Returns:
            tuple or single value depending on `extra_returns`:
                - action
                - the preprocessed states
        """
        if time_percentage is None:
            time_percentage = self.timesteps / (self.max_timesteps or 1e6)

        extra_returns = [extra_returns] if isinstance(extra_returns, str) else (extra_returns or [])
        # States come in without preprocessing -> use state space.
        if apply_preprocessing:
            call_method = "get_actions"
            batched_states, remove_batch_rank = self.state_space.force_batch(states)
        else:
            call_method = "get_actions_from_preprocessed_states"
            batched_states = states
            remove_batch_rank = False  #batched_states.ndim == np.asarray(states).ndim + 1

        # Increase timesteps by the batch size (number of states in batch).
        batch_size = len(batched_states)
        self.timesteps += batch_size

        ret = self.graph_executor.execute((
            call_method,
            [batched_states, not use_exploration, time_percentage],  # `deterministic`=not use_exploration
            # Control, which return value to "pull" (depending on `extra_returns`).
            ["actions"] + list(extra_returns)
        ))

        if remove_batch_rank:
            return strip_list(ret)
        else:
            return ret

    def _observe_graph(self, preprocessed_states, actions, internals, rewards, terminals, **kwargs):
        next_states = kwargs.pop("next_states")
        self.graph_executor.execute(("insert_records", [preprocessed_states, actions, rewards, next_states, terminals]))

    def update(self, batch=None, time_percentage=None, **kwargs):
        if time_percentage is None:
            time_percentage = self.timesteps / (self.max_timesteps or 1e6)

        self.num_updates += 1

        # Should we sync the target net?
        self.steps_since_target_net_sync += 1
        if self.steps_since_target_net_sync >= self.sync_rules.sync_every_n_updates:
            sync_call = "sync_target_qnet"
            self.steps_since_target_net_sync = 0
        else:
            sync_call = None

        if batch is None:
            ret = self.graph_executor.execute(("update_from_memory", [True, time_percentage]))
        else:
            # TODO apply postprocessing always true atm.
            input_ = [
                batch["states"], batch["actions"], batch["rewards"], batch["terminals"], batch["next_states"],
                batch["importance_weights"], True, time_percentage
            ]
            ret = self.graph_executor.execute(("update_from_external_batch", input_))

        # Do the target net synching after the update (for better clarity: after a sync, we would expect for both
        # networks to be the exact same).
        if sync_call:
            self.graph_executor.execute(sync_call)

        return ret["loss"], ret["loss_per_item"]

    def reset(self):
        """
        Resets our preprocessor, but only if it contains stateful PreprocessLayer Components (meaning
        the PreprocessorStack has at least one variable defined).
        """
        if self.root_component.preprocessing_required and len(self.root_component.preprocessor.variable_registry) > 0:
            self.graph_executor.execute("reset_preprocessor")

    def post_process(self, batch):
        batch_input = [batch["states"], batch["actions"], batch["rewards"], batch["terminals"],
                       batch["next_states"], batch["importance_weights"]]
        ret = self.graph_executor.execute(("get_td_loss", batch_input))

        # Remove unnecessary return dicts.
        if isinstance(ret, dict):
            ret = ret["get_td_loss"]

        return ret["loss"], ret["loss_per_item"]

    def __repr__(self):
        return "DQNAgent(doubleQ={} duelingQ={})".format(self.root_component.double_q, self.root_component.dueling_q)


class DQNAlgorithmComponent(AlgorithmComponent):
    def __init__(
            self, agent, *, memory_spec=None, double_q=False, dueling_q=False, n_step=1,
            huber_loss=False, shared_container_action_target=True,
            scope="dqn-agent-component", **kwargs
    ):
        """
        Args:
        """
        # Fix action-adapter before passing it to the super constructor.
        # Use a DuelingPolicy (instead of a basic Policy) if option is set.
        policy_spec = kwargs.pop("policy_spec", None)
        if dueling_q is True:
            if policy_spec is None:
                policy_spec = dict()
            assert isinstance(policy_spec, (dict, DuelingPolicy)),\
                "ERROR: If `dueling_q` is True, policy must be specified as DuelingPolicy!"
            if isinstance(policy_spec, dict):
                policy_spec["type"] = "dueling-policy"
                # Give us some default state-value nodes.
                if "units_state_value_stream" not in policy_spec:
                    policy_spec["units_state_value_stream"] = 128

        super(DQNAlgorithmComponent, self).__init__(agent, policy_spec=policy_spec, scope=scope, **kwargs)

        self.double_q = double_q
        self.dueling_q = dueling_q
        self.n_step = n_step

        # The replay memory.
        self.memory = Memory.from_spec(memory_spec)
        # Make sure the python buffer is not larger than our memory capacity.
        assert not self.agent or self.agent.python_buffer_size <= self.memory.capacity, \
            "ERROR: Python buffer's size ({}) must be smaller or equal to the memory's capacity ({})!". \
            format(self.agent.python_buffer_size, self.memory.capacity)

        # Copy our Policy (target-net), make target-net synchronizable.
        self.target_policy = self.policy.copy(scope="target-policy", trainable=False)
        # Number of steps since the last target-net synching from the main policy.
        self.steps_since_target_net_sync = 0

        use_importance_weights = isinstance(self.memory, PrioritizedReplay)
        self.loss_function = DQNLossFunction(
            discount=self.discount, double_q=self.double_q, huber_loss=huber_loss,
            shared_container_action_target=shared_container_action_target,
            importance_weights=use_importance_weights, n_step=n_step
        )

        # Add our self-created ones.
        self.add_components(self.memory, self.loss_function, self.target_policy)

    def check_input_spaces(self, input_spaces, action_space=None):
        for s in ["states", "actions", "preprocessed_states", "rewards", "terminals"]:
            sanity_check_space(input_spaces[s], must_have_batch_rank=True)

    # Insert into memory.
    @rlgraph_api
    def insert_records(self, preprocessed_states, actions, rewards, next_states, terminals):
        records = dict(
            states=preprocessed_states, actions=actions, rewards=rewards, next_states=next_states, terminals=terminals
        )
        return self.memory.insert_records(records)

    # Syncing target-net.
    @rlgraph_api
    def sync_target_qnet(self):
        # If we are a multi-GPU root:
        # Simply feeds everything into the multi-GPU sync optimizer's method and return.
        if "multi-gpu-synchronizer" in self.sub_components:
            multi_gpu_syncer = self.sub_components["multi-gpu-synchronizer"]
            return multi_gpu_syncer.sync_target_qnets()
        # We could be the main root or a multi-GPU tower.
        else:
            policy_vars = self.policy.variables()
            return self.target_policy.sync(policy_vars)

    # Learn from memory.
    @rlgraph_api
    def update_from_memory(self, apply_postprocessing, time_percentage=None):
        # Non prioritized memory will just return weight 1.0 for all samples.
        records, sample_indices, importance_weights = self.memory.get_records(self.memory_batch_size)

        out = self.update_from_external_batch(
            records["states"], records["actions"], records["rewards"], records["terminals"], records["next_states"],
            importance_weights, apply_postprocessing, time_percentage
        )

        # TODO this is really annoying. Will be solved once we have dict returns.
        if isinstance(self.memory, PrioritizedReplay):
            update_pr_step_op = self.memory.update_records(sample_indices, out["loss_per_item"])
            out["step_op"] = self._graph_fn_group(update_pr_step_op, out["step_op"])
        return dict(step_op=out["step_op"], loss=out["loss"], loss_per_item=out["loss_per_item"])

    # Learn from an external batch.
    @rlgraph_api
    def update_from_external_batch(
            self, preprocessed_states, actions, rewards, terminals, preprocessed_next_states,
            importance_weights, apply_postprocessing, time_percentage=None
    ):
        # If we are a multi-GPU root:
        # Simply feeds everything into the multi-GPU sync optimizer's method and return.
        if "multi-gpu-synchronizer" in self.sub_components:
            #main_policy_vars = self.policy.variables()
            all_vars = dict(policy=self.policy.variables())  #self.vars_merger.merge(main_policy_vars)
            out = self.sub_components["multi-gpu-synchronizer"].calculate_update_from_external_batch(
                all_vars, preprocessed_states, actions, rewards, terminals,
                preprocessed_next_states, importance_weights, apply_postprocessing=apply_postprocessing,
                time_percentage=time_percentage
            )
            #avg_grads_and_vars = self.vars_splitter.call(out["grads_and_vars_by_component"])
            step_op = self.optimizer.apply_gradients(out["grads_and_vars_by_component"]["policy"])
            # Increase the global training step counter.
            step_op = self._graph_fn_training_step(step_op)
            step_and_sync_op = self.sub_components["multi-gpu-synchronizer"].sync_variables_to_towers(
                step_op, all_vars
            )
            #q_values_s = out["additional_return_0"]
            return dict(step_op=step_and_sync_op, loss=out["loss"], loss_per_item=out["loss_per_item"]) #, q_values_s

        # Get the different Q-values.
        q_values_s = self.policy.get_adapter_outputs_and_parameters(preprocessed_states)["adapter_outputs"]
        qt_values_sp = self.target_policy.get_adapter_outputs_and_parameters(preprocessed_next_states)["adapter_outputs"]

        q_values_sp = None
        if self.double_q:
            q_values_sp = self.policy.get_adapter_outputs_and_parameters(preprocessed_next_states)["adapter_outputs"]

        loss, loss_per_item = self.loss_function.loss(
            q_values_s, actions, rewards, terminals, qt_values_sp, q_values_sp, importance_weights
        )

        # Args are passed in again because some device strategies may want to split them to different devices.
        policy_vars = self.policy.variables()

        # TODO: for a fully automated multi-GPU strategy, we would have to make sure that:
        # TODO: - every agent (root_component) has an update_from_external_batch method
        # TODO: - this if check is somehow automated and not necessary anymore (local optimizer must be called with different API-method, not step)
        if hasattr(self, "is_multi_gpu_tower") and self.is_multi_gpu_tower is True:
            grads_and_vars = self.optimizer.calculate_gradients(policy_vars, loss, time_percentage)
            grads_and_vars_by_component = dict(policy=grads_and_vars) # self.vars_merger.merge(grads_and_vars)
            return dict(grads_and_vars_by_component=grads_and_vars_by_component, loss=loss, loss_per_item=loss_per_item)
        else:
            step_op = self.optimizer.step(policy_vars, loss, loss_per_item, time_percentage)
            # Increase the global training step counter.
            step_op = self._graph_fn_training_step(step_op)
            return dict(step_op=step_op, loss=loss, loss_per_item=loss_per_item)

    @rlgraph_api
    def get_td_loss(self, preprocessed_states, actions, rewards,
                    terminals, preprocessed_next_states, importance_weights):

        # Get the different Q-values.
        q_values_s = self.policy.get_adapter_outputs_and_parameters(preprocessed_states)["adapter_outputs"]
        qt_values_sp = self.target_policy.get_adapter_outputs_and_parameters(preprocessed_next_states)["adapter_outputs"]

        q_values_sp = None
        if self.double_q:
            q_values_sp = self.policy.get_adapter_outputs_and_parameters(preprocessed_next_states)["adapter_outputs"]

        loss, loss_per_item = self.loss_function.loss(
            q_values_s, actions, rewards, terminals, qt_values_sp, q_values_sp, importance_weights
        )
        return dict(loss=loss, loss_per_item=loss_per_item)

    @rlgraph_api
    def get_q_values(self, preprocessed_states):
        q_values = self.policy.get_adapter_outputs_and_parameters(
            preprocessed_states)["adapter_outputs"]
        return q_values
