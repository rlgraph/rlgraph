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

import copy
import numpy as np

from yarl.agents import Agent
from yarl.components import Synchronizable, Merger, Splitter, Memory, DQNLossFunction, Policy
from yarl.spaces import Dict, FloatBox, BoolBox
from yarl.utils.visualization_util import get_graph_markup


class DQNAgent(Agent):
    """
    A collection of DQN algorithms published in the following papers:
    [1] Human-level control through deep reinforcement learning. Mnih, Kavukcuoglu, Silver et al. - 2015
    [2] Deep Reinforcement Learning with Double Q-learning. v. Hasselt, Guez, Silver - 2015
    [3] Dueling Network Architectures for Deep Reinforcement Learning, Wang et al. - 2016
    """

    def __init__(self, discount=0.98, memory_spec=None, double_q=True, dueling_q=True, **kwargs):
        """
        Args:
            discount (float): The discount factor (gamma).
            memory_spec (Optional[dict,Memory]): The spec for the Memory to use for the DQN algorithm.
            double_q (bool): Whether to use the double DQN loss function (see [2]).
            dueling_q (bool): Whether to use a dueling layer in the ActionAdapter  (see [3]).
        """
        super(DQNAgent, self).__init__(**kwargs)

        self.discount = discount
        self.memory = Memory.from_spec(memory_spec)
        self.record_space = Dict(states=self.state_space, actions=self.action_space, rewards=float,
                                 terminals=BoolBox(), add_batch_rank=False)
        self.double_q = double_q
        self.dueling_q = dueling_q

        # The target policy (is synced from the q-net policy every n steps).
        self.target_policy = None

        self.policy = Policy(
            neural_network=self.neural_network, action_adapter_spec=dict(add_dueling_layer=self.dueling_q)
        )
        # Copy our Policy (target-net), make target-net synchronizable.
        self.target_policy = self.policy.copy(scope="target-policy")
        self.target_policy.add_components(Synchronizable())

        self.merger = Merger(output_space=self.record_space)
        splitter_input_space = copy.deepcopy(self.record_space)
        splitter_input_space["next_states"] = self.state_space
        self.splitter = Splitter(input_space=splitter_input_space)
        self.loss_function = DQNLossFunction(discount=self.discount, double_q=self.double_q)

        # Define our Spaces.
        state_space = self.state_space.with_batch_rank()
        action_space = self.action_space.with_batch_rank()
        reward_space = FloatBox(add_batch_rank=True)
        terminal_space = BoolBox(add_batch_rank=True)

        # Define our api_methods.
        self.input_spaces = dict(
            get_action=[int, state_space],
            insert_records=[state_space, action_space, reward_space, terminal_space],
            update_from_external_batch=[state_space, state_space, action_space, reward_space, terminal_space]
        )

        self.assemble_meta_graph()
        # markup = get_graph_markup(self.graph_builder.core_component)
        # print(markup)
        self.build_graph(self.input_spaces, self.optimizer)

    def _assemble_meta_graph(self, core, *params):

        # Add all our sub-components to the core.
        core.add_components(self.preprocessor, self.memory, self.merger, self.splitter, self.policy,
                            self.target_policy, self.exploration, self.loss_function, self.optimizer)

        # Define the Agent's API methods.

        # State from environment to action pathway.
        def get_action(self_, states, time_step):
            preprocessed_states = self_.call(self_.preprocessor_stack.preprocess, states)
            sample_deterministic = self_.call(self_.policy.sample_deterministic, preprocessed_states)
            sample_stochastic = self_.call(self_.policy.sample_stochastic, preprocessed_states)
            return self_.call(self_.exploration.get_action(time_step, sample_deterministic, sample_stochastic))

        core.define_api_method("get_action", get_action)

        # Insert into memory pathway.
        def insert_records(self_, states, actions, rewards, terminals):
            preprocessed = self_.call(self_.preprocessor.preprocess, states)
            records = self_.call(self_.merger.merge, preprocessed, actions, rewards, terminals)
            return self_.call(self_.memory.insert_records, records)

        core.define_api_method("insert_records", insert_records)

        # Syncing target-net.
        def synch_target_qnet(self_):
            policy_vars = self_.call(self_.policy._variables)
            return self_.call(self_.target_policy.sync, policy_vars)

        core.define_outputs("sync_target_qnet", synch_target_qnet)

        # Learn from memory.
        def update_from_memory(self_):
            records = self_.call(self_.memory.get_records, self.update_spec["batch_size"])
            states, actions, rewards, terminals, next_states = self_.call(self_.splitter.split, records)
            # Get the different Q-values.
            if self_.dueling_q:
                _, _, q_values_s = self_.call(self_.policy.get_dueling_output, states)
                _, _, qt_values_sp = self_.call(self_.target_policy.get_dueling_output, next_states)
            else:
                q_values_s = self_.call(self_.policy.get_action_layer_output_reshaped, states)
                qt_values_sp = self_.call(self_.target_policy.get_action_layer_output_reshaped, next_states)

            if self_.double_q:
                if self_.dueling_q:
                    _, _, q_values_sp = self_.call(self_.policy.get_dueling_output, next_states)
                else:
                    q_values_sp = self_.call(self_.policy.get_action_layer_output_reshaped, next_states)
                loss = self_.call(self_.loss_function.loss, q_values_s, actions, rewards,
                                  terminals, qt_values_sp, q_values_sp)
            else:
                loss = self_.call(self_.loss_function.loss, q_values_s, actions, rewards,
                                  terminals, qt_values_sp)

            policy_vars = self_.call(self_.policy._variables)
            return self_.call(self_.optimizer.step(policy_vars, loss))

        core.define_api_method("update_from_memory", update_from_memory)

        # Learn from an external batch.
        def update_from_external_batch(self_, states, actions, rewards, terminals, next_states):
            preprocessed_s = self_.call(self_.preprocessor.preprocess, states)
            preprocessed_sp = self_.call(self_.preprocessor.preprocess, next_states)

            # Get the different Q-values.
            if self_.dueling_q:
                _, _, q_values_s = self_.call(self_.policy.get_dueling_output, preprocessed_s)
                _, _, qt_values_sp = self_.call(self_.target_policy.get_dueling_output, preprocessed_sp)
            else:
                q_values_s = self_.call(self_.policy.get_action_layer_output_reshaped, preprocessed_s)
                qt_values_sp = self_.call(self_.target_policy.get_action_layer_output_reshaped, preprocessed_sp)

            if self_.double_q:
                if self_.dueling_q:
                    _, _, q_values_sp = self_.call(self_.policy.get_dueling_output, preprocessed_sp)
                else:
                    q_values_sp = self_.call(self_.policy.get_action_layer_output_reshaped, preprocessed_sp)
                loss_per_item = self_.call(self_.loss_function.loss, q_values_s, actions, rewards, terminals,
                                           qt_values_sp, q_values_sp)
            else:
                loss_per_item = self_.call(self_.loss_function.loss, q_values_s, actions, rewards, terminals,
                                           qt_values_sp)
            return self_.call(self_.optimizer.step(loss_per_item))

        core.define_api_method("update_from_external_batch", update_from_external_batch)

    def get_action(self, states, deterministic=False):
        batched_states = self.state_space.batched(states)
        remove_batch_rank = batched_states.ndim == np.asarray(states).ndim + 1
        # Increase timesteps by the batch size (number of states in batch).
        self.timesteps += len(batched_states)
        actions = self.graph_executor.execute(
            api_methods=dict(get_actions=[batched_states, self.timesteps])
        )
        #print("states={} action={} q_values={} do_explore={}".format(states, actions, q_values, do_explore))
        if remove_batch_rank:
            return actions[0]
        return actions

    def _observe_graph(self, states, actions, internals, rewards, terminals):
        self.graph_executor.execute(api_methods=dict(insert_records=[states, actions, rewards, terminals]))

    def update(self, batch=None):
        # Should we sync the target net? (timesteps-1 b/c it has been increased already in get_action)
        if (self.timesteps - 1) % self.update_spec["sync_interval"] == 0:
            self.graph_executor.execute("sync_target_qnet")
        if batch is None:
            _, loss = self.graph_executor.execute("update_from_memory")
        else:
            batch_input = dict(
                external_batch_states=batch["states"],
                external_batch_actions=batch["actions"],
                external_batch_rewards=batch["rewards"],
                external_batch_terminals=batch["terminals"],
                external_batch_next_states=batch["next_states"]
            )
            _, loss = self.graph_executor.execute(api_methods=dict(update_from_external_batch=batch_input))
        return loss

    def __repr__(self):
        return "DQNAgent(doubleQ={})".format(self.double_q)

