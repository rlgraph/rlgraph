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
from yarl.components import CONNECT_ALL, Synchronizable, Merger, Splitter, Memory, DQNLossFunction, Policy
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
        self.target_policy.add_component(Synchronizable(), connections=CONNECT_ALL)

        self.merger = Merger(output_space=self.record_space)
        splitter_input_space = copy.deepcopy(self.record_space)
        splitter_input_space["next_states"] = self.state_space
        self.splitter = Splitter(input_space=splitter_input_space)
        self.loss_function = DQNLossFunction(discount=self.discount, double_q=self.double_q)

        self.assemble_meta_graph(self.preprocessor_stack, self.memory, self.merger, self.splitter, self.policy,
                                 self.target_policy, self.exploration, self.loss_function, self.optimizer)
        # markup = get_graph_markup(self.graph_builder.core_component)
        # print(markup)
        self.build_graph()

    def _assemble_meta_graph(self, core, *params):
        # Define our interface.
        core.define_inputs("states_from_env", "external_batch_states", "external_batch_next_states",
                           "states_for_memory", space=self.state_space.with_batch_rank())
        core.define_inputs("actions_for_memory", "external_batch_actions", space=self.action_space.with_batch_rank())
        core.define_inputs("rewards_for_memory", "external_batch_rewards", space=FloatBox(add_batch_rank=True))
        core.define_inputs("terminals_for_memory", "external_batch_terminals", space=BoolBox(add_batch_rank=True))

        #core.define_inputs("deterministic", space=bool)
        core.define_inputs("time_step", space=int)
        core.define_outputs("get_actions", "insert_records",
                            "update_from_memory", "update_from_external_batch",
                            "sync_target_qnet", "get_batch", "loss")

        # Add the Q-net, copy it (target-net) and add the target-net.
        self.target_policy = self.policy.copy(scope="target-policy")
        # Make target_policy writable
        self.target_policy.add_component(Synchronizable(), connections=CONNECT_ALL)
        core.add_components(self.policy, self.target_policy)
        # Add an Exploration for the q-net (target-net doesn't need one).
        core.add_components(self.exploration)

        # Add our Memory Component plus merger and splitter.
        core.add_components(self.memory, self.merger, self.splitter)

        # Add the loss function and optimizer.
        core.add_components(self.loss_function, self.optimizer)

        # Now connect everything ...

        # All external/env states into preprocessor (memory already preprocessed).
        core.connect("states_from_env", (self.preprocessor_stack, "input"), label="env,s")
        core.connect("external_batch_states", (self.preprocessor_stack, "input"), label="ext,s")
        core.connect("external_batch_next_states", (self.preprocessor_stack, "input"), label="ext,sp")
        core.connect((self.preprocessor_stack, "output"), (self.policy, "nn_input"),
                     label="s"+(",sp" if self.double_q else ""))

        # Timestep into Exploration.
        core.connect("time_step", (self.exploration, "time_step"))

        # Policy output into Exploration -> into "actions".
        core.connect((self.policy, "sample_deterministic"),
                     (self.exploration, "sample_deterministic"), label="env")
        core.connect((self.policy, "sample_stochastic"),
                     (self.exploration, "sample_stochastic"), label="env")
        core.connect((self.exploration, "action"), "get_actions")
        #core.connect((self.exploration, "do_explore"), "do_explore")

        # Insert records into memory via merger.
        core.connect("states_for_memory", (self.preprocessor_stack, "input"), label="to_mem")
        core.connect((self.preprocessor_stack, "output"), (self.merger, "/states"), label="to_mem")
        for in_ in ["actions", "rewards", "terminals"]:
            core.connect(in_+"_for_memory", (self.merger, "/"+in_))
        core.connect((self.merger, "output"), (self.memory, "records"))
        core.connect((self.memory, "insert_records"), "insert_records")

        # Learn from Memory via get_batch and Splitter.
        core.connect(self.update_spec["batch_size"], (self.memory, "num_records"))
        core.connect((self.memory, "get_records"), (self.splitter, "input"), label="mem")
        core.connect((self.memory, "get_records"), "get_batch")
        core.connect((self.splitter, "/states"), (self.policy, "nn_input"), label="mem,s")
        core.connect((self.splitter, "/actions"), (self.loss_function, "actions"))
        core.connect((self.splitter, "/rewards"), (self.loss_function, "rewards"))
        core.connect((self.splitter, "/terminals"), (self.loss_function, "terminals"))
        core.connect((self.splitter, "/next_states"), (self.target_policy, "nn_input"), label="mem,sp")
        if self.double_q:
            core.connect((self.splitter, "/next_states"), (self.policy, "nn_input"), label="mem,sp")

        # Only send ext and mem labelled ops into loss function.
        q_values_socket = "q_values" if self.dueling_q is True else "action_layer_output_reshaped"
        core.connect((self.policy, q_values_socket), (self.loss_function, "q_values"), label="ext,mem,s")
        #core.connect((self.policy, q_values_socket), "q_values")
        core.connect((self.target_policy, q_values_socket), (self.loss_function, "qt_values_s_"), label="ext,mem")
        if self.double_q:
            core.connect((self.policy, q_values_socket), (self.loss_function, "q_values_s_"), label="ext,mem,sp")

        # Connect the Optimizer.
        core.connect((self.loss_function, "loss"), (self.optimizer, "loss"))
        core.connect((self.loss_function, "loss"), "loss")
        core.connect((self.policy, "_variables"), (self.optimizer, "vars"))
        core.connect((self.optimizer, "step"), "update_from_memory", label="mem")
        core.connect((self.optimizer, "step"), "update_from_external_batch", label="ext")

        # Add syncing capability for target-net.
        core.connect((self.policy, "_variables"), (self.target_policy, "_values"))
        core.connect((self.target_policy, "sync"), "sync_target_qnet")

    def _assemble_meta_graph_test(self, core, preprocessor, memory, merger, splitter, policy, target_policy,
                                  exploration, loss_function, optimizer):
        # Define our Spaces.
        state_space = self.state_space.with_batch_rank()
        action_space = self.action_space.with_batch_rank()
        reward_space = FloatBox(add_batch_rank=True)
        terminal_space = BoolBox(add_batch_rank=True)

        # Define our inputs.
        inputs = dict(
            states_from_env=state_space,

            states_to_memory=state_space,
            actions_to_memory=action_space,
            rewards_to_memory=reward_space,
            terminals_to_memory=terminal_space,

            states_from_external=state_space,
            next_states_from_external=state_space,
            actions_from_external=action_space,
            rewards_from_external=reward_space,
            terminals_from_external=terminal_space,

            time_step=bool,
        )
        core.define_inputs(inputs)
        # Add all sub-components.
        core.add_components(preprocessor, memory, merger, splitter, policy, target_policy, exploration,
                            loss_function, optimizer)

        # Env pathway.
        preprocessed_states_from_env = preprocessor("states_from_env")
        sample_deterministic, sample_stochastic = policy(
            preprocessed_states_from_env, ["sample_deterministic", "sample_stochastic"]
        )
        action = exploration(["time_step", sample_deterministic, sample_stochastic])
        core.define_outputs("get_actions", action)

        # Insert into memory pathway.
        preprocessed_states_to_mem = preprocessor("states_to_memory")
        records = merger([preprocessed_states_to_mem, "actions_to_memory", "rewards_to_memory", "terminals_to_memory"])
        insert_records_op = memory(records, "insert_records")
        core.define_outputs("insert_records", insert_records_op)

        # Syncing target-net.
        policy_vars = policy(None, "_variables")
        sync_op = target_policy(policy_vars, "sync")
        core.define_outputs("sync_target_qnet", sync_op)

        # Learn from memory.
        q_values_socket_name = "q_values" if self.dueling_q is True else "action_layer_output_reshaped"
        records_from_memory = memory(self.update_spec["batch_size"], "get_records")
        s_mem, a_mem, r_mem, t_mem, sp_mem = splitter(records_from_memory)
        q_values_s = policy(s_mem, q_values_socket_name)
        qt_values_sp = target_policy(sp_mem, q_values_socket_name)
        if self.double_q:
            q_values_sp = policy(sp_mem, q_values_socket_name)
            loss_per_item = loss_function([q_values_s, a_mem, r_mem, t_mem, qt_values_sp, q_values_sp], "loss_per_item")
        else:
            loss_per_item = loss_function([q_values_s, a_mem, r_mem, t_mem, qt_values_sp], "loss_per_item")
        update_from_mem = optimizer(loss_per_item)
        optimizer(policy_vars, None)  # TODO: this will probably not work
        core.define_outputs("update_from_memory", update_from_mem)

        # Learn from external batch.
        preprocessed_s_from_external = preprocessor("states_from_external")
        preprocessed_sp_from_external = preprocessor("next_states_from_external")
        q_values_s = policy(preprocessed_s_from_external, q_values_socket_name)
        qt_values_sp = target_policy(preprocessed_sp_from_external, q_values_socket_name)
        if self.double_q:
            q_values_sp = policy(preprocessed_sp_from_external, q_values_socket_name)
            loss_per_item = loss_function([q_values_s, "actions_from_external", "rewards_from_external",
                                           "terminals_from_external", qt_values_sp, q_values_sp], "loss_per_item")
        else:
            loss_per_item = loss_function([q_values_s, "actions_from_external", "rewards_from_external",
                                           "terminals_from_external", qt_values_sp], "loss_per_item")
        update_from_external = optimizer(loss_per_item)
        core.define_outputs("update_from_external_batch", update_from_external)

    def get_action(self, states, deterministic=False):
        batched_states = self.state_space.batched(states)
        remove_batch_rank = batched_states.ndim == np.asarray(states).ndim + 1
        # Increase timesteps by the batch size (number of states in batch).
        self.timesteps += len(batched_states)
        actions = self.graph_executor.execute(
            "get_actions", inputs=dict(states_from_env=batched_states, time_step=self.timesteps)
        )
        #print("states={} action={} q_values={} do_explore={}".format(states, actions, q_values, do_explore))
        if remove_batch_rank:
            return actions[0]
        return actions

    def _observe_graph(self, states, actions, internals, rewards, terminals):
        self.graph_executor.execute("insert_records", inputs=dict(
            states_for_memory=states,
            actions_for_memory=actions,
            rewards_for_memory=rewards,
            terminals_for_memory=terminals
        ))

    def update(self, batch=None):
        # Should we sync the target net? (timesteps-1 b/c it has been increased already in get_action)
        if (self.timesteps - 1) % self.update_spec["sync_interval"] == 0:
            self.graph_executor.execute("sync_target_qnet")
        if batch is None:
            _, loss = self.graph_executor.execute(["update_from_memory", "loss"])
        else:
            batch_input = dict(
                external_batch_states=batch["states"],
                external_batch_actions=batch["actions"],
                external_batch_rewards=batch["rewards"],
                external_batch_terminals=batch["terminals"],
                external_batch_next_states=batch["next_states"]
            )
            _, loss = self.graph_executor.execute(
                ["update_from_external_batch", "loss"], inputs=batch_input
            )
        return loss

    def __repr__(self):
        return "DQNAgent(doubleQ={})".format(self.double_q)

