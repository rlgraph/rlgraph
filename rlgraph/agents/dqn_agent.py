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

import numpy as np

from rlgraph.agents import Agent
from rlgraph.components import Synchronizable, Memory, PrioritizedReplay, DQNLossFunction, Policy, Merger, Splitter
from rlgraph.spaces import FloatBox, BoolBox
from rlgraph.utils.util import strip_list


class DQNAgent(Agent):
    """
    A collection of DQN algorithms published in the following papers:
    [1] Human-level control through deep reinforcement learning. Mnih, Kavukcuoglu, Silver et al. - 2015
    [2] Deep Reinforcement Learning with Double Q-learning. v. Hasselt, Guez, Silver - 2015
    [3] Dueling Network Architectures for Deep Reinforcement Learning, Wang et al. - 2016
    [4] https://en.wikipedia.org/wiki/Huber_loss
    """

    def __init__(self, double_q=True, dueling_q=True, huber_loss=False, memory_spec=None,
                 store_last_memory_batch=False, store_last_q_table=False, **kwargs):
        """
        Args:
            double_q (bool): Whether to use the double DQN loss function (see [2]).
            dueling_q (bool): Whether to use a dueling layer in the ActionAdapter  (see [3]).
            huber_loss (bool) : Whether to apply a Huber loss. (see [4]).
            memory_spec (Optional[dict,Memory]): The spec for the Memory to use for the DQN algorithm.
            store_last_memory_batch (bool): Whether to store the last pulled batch from the memory in
                `self.last_memory_batch` for debugging purposes.
                Default: False.
            store_last_q_table (bool): Whether to store the Q(s,a) values for the last received batch
                (memory or external) in `self.last_q_table` for debugging purposes.
                Default: False.
        """
        # Fix action-adapter before passing it to the super constructor.
        action_adapter_spec = kwargs.pop("action_adapter_spec", dict())
        action_adapter_spec.update(dict(add_dueling_layer=dueling_q))
        super(DQNAgent, self).__init__(
            action_adapter_spec=action_adapter_spec, name=kwargs.pop("name", "dqn-agent"), **kwargs
        )

        self.double_q = double_q
        self.dueling_q = dueling_q
        self.huber_loss = huber_loss

        # Debugging tools.
        self.store_last_memory_batch = store_last_memory_batch
        self.last_memory_batch = None
        self.store_last_q_table = store_last_q_table
        self.last_q_table = None

        # Extend input Space definitions to this Agent's specific API-methods.
        state_space = self.state_space.with_batch_rank()
        preprocessed_state_space = self.preprocessed_state_space.with_batch_rank()
        action_space = self.action_space.with_batch_rank()
        reward_space = FloatBox(add_batch_rank=True)
        terminal_space = BoolBox(add_batch_rank=True)
        weight_space = FloatBox(add_batch_rank=True)
        self.input_spaces.update(dict(
            get_preprocessed_state_and_action=[state_space, int, bool],  # state, time-step, use_exploration
            insert_records=[preprocessed_state_space, action_space, reward_space, terminal_space],
            update_from_external_batch=[preprocessed_state_space, action_space, reward_space,
                                        terminal_space, preprocessed_state_space, weight_space]
        ))

        # The merger to merge inputs into one record Dict going into the memory.
        self.merger = Merger("states", "actions", "rewards", "terminals")
        # The replay memory.
        self.memory = Memory.from_spec(memory_spec)
        # The splitter for splitting up the records coming from the memory.
        self.splitter = Splitter("states", "actions", "rewards", "terminals", "next_states")

        # Copy our Policy (target-net), make target-net synchronizable.
        self.target_policy = self.policy.copy(scope="target-policy", trainable=False)
        self.target_policy.add_components(Synchronizable(), expose_apis="sync")

        use_importance_weights = isinstance(self.memory, PrioritizedReplay)
        self.loss_function = DQNLossFunction(
            discount=self.discount, double_q=self.double_q, huber_loss=self.huber_loss,
            importance_weights=use_importance_weights
        )

        # Add all our sub-components to the core.
        sub_components = [self.preprocessor, self.merger, self.memory, self.splitter, self.policy,
                          self.target_policy, self.exploration, self.loss_function, self.optimizer]
        self.core_component.add_components(*sub_components)

        # Define the Agent's (core Component's) API.
        self.define_api_methods(*sub_components)

        # markup = get_graph_markup(self.graph_builder.core_component)
        # print(markup)
        self.build_graph(self.input_spaces, self.optimizer)

    def define_api_methods(self, preprocessor, merger, memory, splitter, policy, target_policy, exploration,
                           loss_function, optimizer):
        super(DQNAgent, self).define_api_methods()

        # Reset operation (resets preprocessor).
        def reset_preprocessor(self_):
            reset_op = self_.call(preprocessor.reset)
            return reset_op

        self.core_component.define_api_method("reset_preprocessor", reset_preprocessor)

        # State (from environment) to action.
        def get_preprocessed_state_and_action(self_, states, time_step, use_exploration=True):
            preprocessed_states = self_.call(preprocessor.preprocess, states)
            sample_deterministic = self_.call(policy.sample_deterministic, preprocessed_states)
            sample_stochastic = self_.call(policy.sample_stochastic, preprocessed_states)
            actions = self_.call(exploration.get_action, sample_deterministic, sample_stochastic,
                                 time_step, use_exploration)
            # TODO: Alternatively, move exploration (especially epsilon-based) into python.
            # TODO: return internal states as well and maybe the exploration decision
            return preprocessed_states, actions

        self.core_component.define_api_method("get_preprocessed_state_and_action", get_preprocessed_state_and_action)

        # Insert into memory.
        def insert_records(self_, preprocessed_states, actions, rewards, terminals):
            records = self_.call(merger.merge, preprocessed_states, actions, rewards, terminals)
            return self_.call(memory.insert_records, records)

        self.core_component.define_api_method("insert_records", insert_records)

        # Syncing target-net.
        def sync_target_qnet(self_):
            policy_vars = self_.call(policy._variables)
            return self_.call(target_policy.sync, policy_vars)

        self.core_component.define_api_method("sync_target_qnet", sync_target_qnet)

        # Learn from memory.
        def update_from_memory(self_):
            # PRs also return updated weights and their indices.
            sample_indices = None
            importance_weights = None
            if isinstance(memory, PrioritizedReplay):
                records, sample_indices, importance_weights = self_.call(
                    memory.get_records, self.update_spec["batch_size"]
                )
            # Non-PR memory.
            else:
                records = self_.call(memory.get_records, self.update_spec["batch_size"])

            preprocessed_s, actions, rewards, terminals, preprocessed_s_prime = self_.call(splitter.split, records)

            # Get the different Q-values.
            q_values_s = self_.call(policy.get_q_values, preprocessed_s)
            qt_values_sp = self_.call(target_policy.get_q_values, preprocessed_s_prime)
            q_values_sp = None
            if self.double_q:
                q_values_sp = self_.call(policy.get_q_values, preprocessed_s_prime)

            if isinstance(memory, PrioritizedReplay):
                loss, loss_per_item = self_.call(loss_function.loss, q_values_s, actions, rewards, terminals,
                                                 qt_values_sp, q_values_sp, importance_weights)
            else:
                loss, loss_per_item = self_.call(loss_function.loss, q_values_s, actions, rewards, terminals,
                                                 qt_values_sp, q_values_sp)

            policy_vars = self_.call(policy._variables)
            step_op = self_.call(optimizer.step, policy_vars, loss)

            # TODO: For multi-GPU, the final-loss will probably have to come from the optimizer.
            # If we are using a PR: Update its weights based on the loss.
            if isinstance(memory, PrioritizedReplay):
                update_pr_step_op = self_.call(memory.update_records, sample_indices, loss_per_item)
                return step_op, loss, loss_per_item, records, q_values_s, update_pr_step_op
            else:
                return step_op, loss, loss_per_item, records, q_values_s

        self.core_component.define_api_method("update_from_memory", update_from_memory)

        # Learn from an external batch.
        def update_from_external_batch(self_, preprocessed_s, actions, rewards, terminals,
                                       preprocessed_s_prime, importance_weights):
            # Get the different Q-values.
            q_values_s = self_.call(policy.get_q_values, preprocessed_s)
            qt_values_sp = self_.call(target_policy.get_q_values, preprocessed_s_prime)
            q_values_sp = None
            if self.double_q:
                q_values_sp = self_.call(policy.get_q_values, preprocessed_s_prime)

            if isinstance(memory, PrioritizedReplay):
                loss, loss_per_item = self_.call(loss_function.loss, q_values_s, actions, rewards, terminals,
                                     qt_values_sp, q_values_sp, importance_weights)
            else:
                loss, loss_per_item = self_.call(loss_function.loss, q_values_s, actions, rewards, terminals,
                                     qt_values_sp, q_values_sp)

            policy_vars = self_.call(policy._variables)
            step_op = self_.call(optimizer.step, policy_vars, loss)
            # TODO: For multi-GPU, the final-loss will probably have to come from the optimizer.
            return step_op, loss, loss_per_item, q_values_s

        self.core_component.define_api_method("update_from_external_batch", update_from_external_batch)

    def get_action(self, states, internals=None, use_exploration=True, extra_returns=None):
        """
        Args:
            extra_returns (Optional[Set[str],str]): Optional string or set of strings for additional return
                values (besides the actions). Possible values are:
                - 'preprocessed_states': The preprocessed states after passing the given states
                    through the preprocessor stack.
                - 'internal_states': The internal states returned by the RNNs in the NN pipeline.
                - 'used_exploration': Whether epsilon- or noise-based exploration was used or not.

        Returns:
            tuple or single value depending on `extra_returns`:
                - action
                - the preprocessed states
        """
        extra_returns = {extra_returns} if isinstance(extra_returns, str) else (extra_returns or set())

        batched_states = self.state_space.force_batch(states)
        remove_batch_rank = batched_states.ndim == np.asarray(states).ndim + 1

        # Increase timesteps by the batch size (number of states in batch).
        self.timesteps += len(batched_states)

        # Control, which return value to "pull" (depending on `additional_returns`).
        return_ops = [1, 0] if "preprocessed_states" in extra_returns else [1]
        ret = self.graph_executor.execute((
            "get_preprocessed_state_and_action",
            [batched_states, self.timesteps, use_exploration],
            # 0=preprocessed_states, 1=action
            return_ops
        ))
        if remove_batch_rank:
            return strip_list(ret)
        else:
            return ret

    def _observe_graph(self, preprocessed_states, actions, internals, rewards, terminals):
        self.graph_executor.execute(("insert_records", [preprocessed_states, actions, rewards, terminals]))

    def update(self, batch=None):
        # Should we sync the target net? (timesteps-1 b/c it has been increased already in get_action)
        if self.timesteps % self.update_spec["sync_interval"] == 0:
            sync_call = "sync_target_qnet"
        else:
            sync_call = None

        # [0]=no-op step; [1]=the loss; [2]=loss-per-item, [3]=memory-batch (if pulled); [4]=q-values
        return_ops = [0, 1, 2]
        q_table = None

        if batch is None:
            # Add some additional return-ops to pull (left out normally for performance reasons).
            if self.store_last_q_table is True:
                return_ops += [3, 4]  # 3=batch, 4=q-values
            elif self.store_last_memory_batch is True:
                return_ops += [3]  # 3=batch
            ret = self.graph_executor.execute(("update_from_memory", None, return_ops), sync_call)

            # Remove unnecessary return dicts (e.g. sync-op).
            if isinstance(ret, dict):
                ret = ret["update_from_memory"]

            # Store the last Q-table?
            if self.store_last_q_table is True:
                q_table = dict(
                    states=ret[3]["states"],
                    q_values=ret[4]
                )
        else:
            # Add some additional return-ops to pull (left out normally for performance reasons).
            if self.store_last_q_table is True:
                return_ops += [3]  # 3=q-values

            batch_input = [batch["states"], batch["actions"], batch["rewards"], batch["terminals"],
                           batch["next_states"], batch["importance_weights"]]
            ret = self.graph_executor.execute(("update_from_external_batch", batch_input, return_ops), sync_call)

            # Remove unnecessary return dicts (e.g. sync-op).
            if isinstance(ret, dict):
                ret = ret["update_from_external_batch"]

            # Store the last Q-table?
            if self.store_last_q_table is True:
                q_table = dict(
                    states=batch["states"],
                    q_values=ret[3]
                )

        # Store the latest pulled memory batch?
        if self.store_last_memory_batch is True and batch is None:
            self.last_memory_batch = ret[2]
        if self.store_last_q_table is True:
            self.last_q_table = q_table

        # [1]=the loss (0=update noop)
        # [2]=loss per item for external update, records for update from memory
        return ret[1], ret[2]

    def reset(self):
        """
        Resets our preprocessor, but only if it contains stateful PreprocessLayer Components (meaning
        the PreprocessorStack has at least one variable defined).
        """
        if len(self.preprocessor.variables) > 0:
            self.graph_executor.execute("reset_preprocessor")

    def __repr__(self):
        return "DQNAgent(doubleQ={} duelingQ={})".format(self.double_q, self.dueling_q)
