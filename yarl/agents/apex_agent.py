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
from yarl.components import CONNECT_ALL, Synchronizable, Merger, Splitter, Memory, DQNLossFunction
from yarl.spaces import Dict, IntBox, FloatBox
from yarl.utils.visualization_util import get_graph_markup


class ApexAgent(Agent):
    """
    Ape-X is a DQN variant designed for large scale distributed execution where many workers
    share a distributed prioritized experience replay.

    Paper: https://arxiv.org/abs/1803.00933

    The distinction to standard DQN is mainly that Ape-X needs to provide additional operations
    to enable external updates of priorities. Ape-X also enables per default dueling and double
    DQN.
    """

    def __init__(self, discount=0.98, memory_spec=None, **kwargs):
        """
        Args:
            discount (float): The discount factor (gamma).
            memory_spec (Optional[dict,Memory]): The spec for the Memory to use for the DQN algorithm.
        """
        super(ApexAgent, self).__init__(**kwargs)

        self.discount = discount
        self.memory = Memory.from_spec(memory_spec)
        self.record_space = Dict(states=self.state_space, actions=self.action_space, rewards=float,
                                 terminals=IntBox(1), add_batch_rank=False)

        # The target policy (is synced from the q-net policy every n steps).
        self.target_policy = None
        # The global copy of the q-net (if we are running in distributed mode).
        self.global_qnet = None

        # TODO: "states_preprocessed" in-Socket + connect properly
        self.merger = Merger(output_space=self.record_space)
        splitter_input_space = copy.deepcopy(self.record_space)
        splitter_input_space["next_states"] = self.state_space
        self.splitter = Splitter(input_space=splitter_input_space)
        self.loss_function = DQNLossFunction(double_q=True)

        self.assemble_meta_graph()
        markup = get_graph_markup(self.graph_builder.core_component)
        self.build_graph()

    def assemble_meta_graph(self):
        core = self.graph_builder.core_component

        # Define our interface.
        core.define_inputs("states", space=self.state_space.with_batch_rank())
        core.define_inputs("actions", space=self.action_space.with_batch_rank())
        core.define_inputs("rewards", space=FloatBox(add_batch_rank=True))
        core.define_inputs("terminals", space=IntBox(2, add_batch_rank=True))
        core.define_inputs("deterministic", space=bool)
        core.define_inputs("time_step", space=int)
        core.define_outputs("get_actions", "insert_records", "update", "sync_target_qnet",
                            # for debugging purposes:
                            "q_values", "loss", "memory_states", "memory_actions", "memory_rewards", "memory_terminals",
                            "do_explore")

        # Add the Q-net, copy it (target-net) and add the target-net.
        self.target_policy = self.policy.copy(scope="target-policy")
        # Make target_policy writable
        self.target_policy.add_component(Synchronizable(), connections=CONNECT_ALL)
        core.add_components(self.target_policy)
        # Add an Exploration for the q-net (target-net doesn't need one).
        core.add_components(self.exploration)

        # Add our Memory Component plus merger and splitter.
        core.add_components(self.memory, self.merger, self.splitter)

        # Add the loss function and optimizer.
        core.add_components(self.loss_function, self.optimizer)

        # Now connect everything ...

        # States (from Env) into preprocessor -> into q-net
        core.connect("states", (self.preprocessor_stack, "input"))
        core.connect((self.preprocessor_stack, "output"), (self.policy, "nn_input"), label="from_env")

        # timestep into Exploration.
        core.connect("time_step", (self.exploration, "time_step"))

        # Network output into Exploration's "nn_output" Socket -> into "actions".
        core.connect((self.policy, "sample_deterministic"),
                     (self.exploration, "sample_deterministic"), label="from_env")
        core.connect((self.policy, "sample_stochastic"),
                     (self.exploration, "sample_stochastic"), label="from_env")
        core.connect((self.exploration, "action"), "get_actions")
        core.connect((self.exploration, "do_explore"), "do_explore")

        # Actions, rewards, terminals into Merger.
        for in_ in ["actions", "rewards", "terminals"]:
            core.connect(in_, (self.merger, "/"+in_))
        # Preprocessed states into Merger.
        core.connect((self.preprocessor_stack, "output"), (self.merger, "/states"))
        # Merger's "output" into Memory's "records".
        core.connect((self.merger, "output"), (self.memory, "records"))
        # Memory's "add_records" functionality.
        core.connect((self.memory, "insert_records"), "insert_records")

        # Memory's "get_records" (to Splitter) and "num_records" (constant batch-size value).
        core.connect((self.memory, "get_records"), (self.splitter, "input"))
        core.connect(self.update_spec["batch_size"], (self.memory, "num_records"))

        # Splitter's outputs.
        core.connect((self.splitter, "/states"), (self.policy, "nn_input"), label="s_from_memory")  # label s from mem
        core.connect((self.splitter, "/states"), "memory_states")
        core.connect((self.splitter, "/actions"), (self.loss_function, "actions"))
        core.connect((self.splitter, "/actions"), "memory_actions")
        core.connect((self.splitter, "/rewards"), (self.loss_function, "rewards"))
        core.connect((self.splitter, "/rewards"), "memory_rewards")
        core.connect((self.splitter, "/terminals"), (self.loss_function, "terminals"))
        core.connect((self.splitter, "/terminals"), "memory_terminals")
        core.connect((self.splitter, "/next_states"), (self.target_policy, "nn_input"))
        core.connect((self.splitter, "/next_states"), (self.policy, "nn_input"), label="sp_from_memory")

        # Loss-function needs both q-values (qnet and target).
        core.connect((self.policy, "logits"), (self.loss_function, "q_values"), label="s_from_memory")
        core.connect((self.policy, "logits"), "q_values")
        core.connect((self.target_policy, "logits"), (self.loss_function, "qt_values_s_"))
        core.connect((self.policy, "logits"), (self.loss_function, "q_values_s_"), label="sp_from_memory")

        # Connect the Optimizer.
        core.connect((self.loss_function, "loss"), (self.optimizer, "loss"))
        core.connect((self.loss_function, "loss"), "loss")
        core.connect((self.policy, "_variables"), (self.optimizer, "vars"))
        core.connect((self.optimizer, "step"), "update")

        # Add syncing capability for target-net.
        core.connect((self.policy, "_variables"), (self.target_policy, "_values"))
        core.connect((self.target_policy, "sync"), "sync_target_qnet")

    def get_action(self, states, deterministic=False):
        batched_states = self.state_space.batched(states)
        remove_batch_rank = batched_states.ndim == np.asarray(states).ndim + 1
        self.timesteps += 1
        actions, q_values, do_explore = self.graph_executor.execute(
            ["get_actions", "q_values", "do_explore"], inputs=dict(states=batched_states, time_step=self.timesteps)
        )
        #print("states={} action={} q_values={} do_explore={}".format(states, actions, q_values, do_explore))
        if remove_batch_rank:
            return actions[0]
        return actions

    def _observe_graph(self, states, actions, internals, rewards, terminals):
        self.graph_executor.execute("insert_records", inputs=dict(
            states=states,
            actions=actions,
            rewards=rewards,
            terminals=terminals
        ))

    def update(self, batch=None):
        # Should we sync the target net? (timesteps-1 b/c it has been increased already in get_action)
        if (self.timesteps - 1) % self.update_spec["sync_interval"] == 0:
            self.graph_executor.execute("sync_target_qnet")
        _, loss, s_, a_, r_, t_ = self.graph_executor.execute(["update", "loss", "memory_states", "memory_actions", "memory_rewards", "memory_terminals"])
        return loss, s_, a_, r_, t_

    def __repr__(self):
        return "ApexAgent"
