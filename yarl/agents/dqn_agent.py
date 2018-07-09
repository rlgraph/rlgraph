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

import numpy as np

from yarl.agents import Agent
from yarl.components import Synchronizable, Memory, DQNLossFunction, Policy, Merger, Splitter
from yarl.spaces import Dict, FloatBox, BoolBox
from yarl.utils.util import strip_list
from yarl.utils.visualization_util import get_graph_markup


class DQNAgent(Agent):
    """
    A collection of DQN algorithms published in the following papers:
    [1] Human-level control through deep reinforcement learning. Mnih, Kavukcuoglu, Silver et al. - 2015
    [2] Deep Reinforcement Learning with Double Q-learning. v. Hasselt, Guez, Silver - 2015
    [3] Dueling Network Architectures for Deep Reinforcement Learning, Wang et al. - 2016
    """

    def __init__(self, discount=0.98, double_q=True, dueling_q=True, memory_spec=None,
                 return_preprocessed_states=False, **kwargs):
        """
        Args:
            discount (float): The discount factor (gamma).
            double_q (bool): Whether to use the double DQN loss function (see [2]).
            dueling_q (bool): Whether to use a dueling layer in the ActionAdapter  (see [3]).
            memory_spec (Optional[dict,Memory]): The spec for the Memory to use for the DQN algorithm.
            return_preprocessed_states (bool): Whether to return (and expect in external-batches) always the already
                preprocessed states (rather than the original states from the env).
                Exception: When calling `Agent.get_action()`, we always expect the original states from the env.
        """
        super(DQNAgent, self).__init__(**kwargs)

        self.discount = discount
        self.double_q = double_q
        self.dueling_q = dueling_q

        # Define the input Space to our API-methods (all batched).
        state_space = self.state_space.with_batch_rank()
        action_space = self.action_space.with_batch_rank()
        reward_space = FloatBox(add_batch_rank=True)
        terminal_space = BoolBox(add_batch_rank=True)
        self.input_spaces = dict(
            get_preprocessed_state_and_action=[state_space, int, bool],
            insert_records=[state_space, action_space, reward_space, terminal_space],
            update_from_external_batch=[state_space, action_space, reward_space, terminal_space, state_space]
        )
        # The merger to merge inputs into one record Dict going into the memory.
        self.merger = Merger("states", "actions", "rewards", "terminals")
        # The replay memory.
        self.memory = Memory.from_spec(memory_spec)
        # The splitter for splitting up the records coming from the memory.
        self.splitter = Splitter("states", "actions", "rewards", "terminals", "next_states")

        # The behavioral policy of the algorithm. Also the one that gets updated.
        self.policy = Policy(
            neural_network=self.neural_network,
            action_adapter_spec=dict(action_space=self.action_space, add_dueling_layer=self.dueling_q)
        )
        # Copy our Policy (target-net), make target-net synchronizable.
        self.target_policy = self.policy.copy(scope="target-policy")
        self.target_policy.add_components(Synchronizable(), expose_apis="sync")

        self.loss_function = DQNLossFunction(discount=self.discount, double_q=self.double_q)

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

        # State from environment to action pathway.
        def get_preprocessed_state_and_action(self_, states, time_step, use_exploration=True):
            preprocessed_states = self_.call(preprocessor.preprocess, states)
            sample_deterministic = self_.call(policy.sample_deterministic, preprocessed_states)
            sample_stochastic = self_.call(policy.sample_stochastic, preprocessed_states)
            actions = self_.call(exploration.get_action, sample_deterministic, sample_stochastic,
                                 time_step, use_exploration)
            return preprocessed_states, actions

        self.core_component.define_api_method("get_preprocessed_state_and_action", get_preprocessed_state_and_action)

        # Insert into memory pathway.
        def insert_records(self_, states, actions, rewards, terminals):
            preprocessed_states = self_.call(preprocessor.preprocess, states)
            records = self_.call(merger.merge, preprocessed_states, actions, rewards, terminals)
            return self_.call(memory.insert_records, records)

        self.core_component.define_api_method("insert_records", insert_records)

        # Syncing target-net.
        def synch_target_qnet(self_):
            policy_vars = self_.call(policy._variables)
            return self_.call(target_policy.sync, policy_vars)

        self.core_component.define_api_method("sync_target_qnet", synch_target_qnet)

        # Learn from memory.
        def update_from_memory(self_):
            records = self_.call(memory.get_records, self.update_spec["batch_size"])

            states, actions, rewards, terminals, next_states = self_.call(splitter.split, records)

            # Get the different Q-values.
            q_values_s = self_.call(policy.get_q_values, states)
            qt_values_sp = self_.call(target_policy.get_q_values, next_states)
            q_values_sp = None
            if self.double_q:
                q_values_sp = self_.call(policy.get_q_values, next_states)

            loss = self_.call(loss_function.loss, q_values_s, actions, rewards, terminals,
                              qt_values_sp, q_values_sp)

            policy_vars = self_.call(policy._variables)
            return self_.call(optimizer.step, policy_vars, loss), loss  # TODO: For multi-GPU, the final-loss will probably have to come from the optimizer.

        self.core_component.define_api_method("update_from_memory", update_from_memory)

        # Learn from an external batch.
        def update_from_external_batch(self_, states, actions, rewards, terminals, next_states):
            preprocessed_s = self_.call(preprocessor.preprocess, states)
            preprocessed_sp = self_.call(preprocessor.preprocess, next_states)

            # Get the different Q-values.
            q_values_s = self_.call(policy.get_q_values, preprocessed_s)
            qt_values_sp = self_.call(target_policy.get_q_values, preprocessed_sp)
            q_values_sp = None
            if self.double_q:
                q_values_sp = self_.call(policy.get_q_values, preprocessed_sp)

            loss = self_.call(loss_function.loss, q_values_s, actions, rewards, terminals,
                              qt_values_sp, q_values_sp)

            policy_vars = self_.call(policy._variables)
            return self_.call(optimizer.step, policy_vars, loss), loss  # TODO:For multi-GPU, the final-loss will probably have to come from the optimizer.

        self.core_component.define_api_method("update_from_external_batch", update_from_external_batch)

    def get_action(self, states, use_exploration=True, return_preprocessed_states=False):
        batched_states = self.state_space.batched(states)
        remove_batch_rank = batched_states.ndim == np.asarray(states).ndim + 1
        # Increase timesteps by the batch size (number of states in batch).
        self.timesteps += len(batched_states)
        preprocessed_states, actions = self.graph_executor.execute(
            api_methods=dict(get_preprocessed_state_and_action=[batched_states, self.timesteps, use_exploration])
        )
        if return_preprocessed_states:
            if remove_batch_rank:
                return strip_list(actions), strip_list(preprocessed_states)
            else:
                return actions, preprocessed_states
        elif remove_batch_rank:
            return strip_list(actions)
        else:
            return actions

    def _observe_graph(self, states, actions, internals, rewards, terminals):
        self.graph_executor.execute(api_methods=dict(insert_records=[states, actions, rewards, terminals]))

    def update(self, batch=None):
        # Should we sync the target net? (timesteps-1 b/c it has been increased already in get_action)
        sync_dict = dict()
        if (self.timesteps - 1) % self.update_spec["sync_interval"] == 0:
            sync_dict["sync_target_qnet"] = None

        if batch is None:
            ret = self.graph_executor.execute(api_methods=dict(update_from_memory=None).update(sync_dict))
            return ret["update_from_memory"][1] if isinstance(ret, dict) else ret[1]  # [1]=the loss (0=update no-op)
        else:
            batch_input = dict(
                external_batch_states=batch["states"],
                external_batch_actions=batch["actions"],
                external_batch_rewards=batch["rewards"],
                external_batch_terminals=batch["terminals"],
                external_batch_next_states=batch["next_states"]
            )
            ret = self.graph_executor.execute(api_methods=dict(update_from_external_batch=batch_input))
            # [1]=the loss (0=update noop)
            return ret["update_from_external_batch"][1] if isinstance(ret, dict) else ret[1]

    def __repr__(self):
        return "DQNAgent(doubleQ={} duelingQ={})".format(self.double_q, self.dueling_q)

