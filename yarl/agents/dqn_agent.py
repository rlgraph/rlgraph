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

from yarl.agents import Agent
from yarl.components.common import Merger, Splitter
from yarl.components.memories import Memory
from yarl.components.loss_functions import DQNLossFunction
from yarl.spaces import Dict, IntBox, FloatBox, BoolBox
from yarl.utils.visualization_util import get_graph_markup


class DQNAgent(Agent):
    """
    A collection of DQN algorithms published in the following papers:
    [1] Human-level control through deep reinforcement learning. Mnih, Kavukcuoglu, Silver et al. - 2015
    [2] Deep Reinforcement Learning with Double Q-learning. v. Hasselt, Guez, Silver - 2015
    [3] Dueling Network Architectures for Deep Reinforcement Learning, Wang et al. - 2016
    """

    def __init__(self, discount=0.98, memory_spec=None, double_q=False, duelling_q=False, **kwargs):
        """
        Args:
            discount (float): The discount factor (gamma).
            memory_spec (Optional[dict,Memory]): The spec for the Memory to use for the DQN algorithm.
            double_q (bool): Whether to use the double DQN loss function (see [2]).
            duelling_q (bool): Whether to use a duelling DQN setup (see [3]).
        """
        super(DQNAgent, self).__init__(**kwargs)

        self.discount = discount
        self.memory = Memory.from_spec(memory_spec)
        self.record_space = Dict(states=self.state_space, actions=self.action_space, rewards=float, terminals=IntBox(1),
                                 add_batch_rank=False)
        self.double_q = double_q
        self.duelling_q = duelling_q

        # The target policy (is synced from the q-net policy every n steps).
        self.target_policy = None
        # The global copy of the q-net (if we are running in distributed mode).
        self.global_qnet = None

        # TODO: "states_preprocessed" in-Socket + connect properly
        self.input_names = ["states", "actions", "rewards", "terminals", "deterministic"]
        self.merger = Merger(output_space=self.record_space)
        splitter_input_space = copy.deepcopy(self.record_space)
        splitter_input_space["next_states"] = self.state_space
        self.splitter = Splitter(input_space=splitter_input_space)
        self.loss_function = DQNLossFunction(double_q=self.double_q)

        self.assemble_meta_graph()
        markup = get_graph_markup(self.graph_builder.core_component)
        print(markup)
        self.compile_graph()

    def assemble_meta_graph(self):
        core = self.graph_builder.core_component

        # Define our interface.
        core.define_inputs(*self.input_names)
        core.define_outputs("act", "add_records", "reset_memory", "learn", "sync_target_qnet")

        # Add the Q-net, copy it (target-net) and add the target-net.
        self.target_policy = self.policy.copy(scope="target-policy")
        core.add_components(self.target_policy)
        # Add an Exploration for the q-net (target-net doesn't need one).
        core.add_components(self.exploration)

        # If we are in distributed mode, add a global qnet as well.
        #if self.execution_spec["mode"] == "distributed":
        #    self.global_qnet = self.neural_network.copy(scope="global-qnet", global_component=True)
        #    core.add_components(self.global_qnet)

        # Add our Memory Component plus merger and splitter.
        core.add_components(self.memory, self.merger, self.splitter)

        # Add the loss function and optimizer.
        core.add_components(self.loss_function, self.optimizer)

        # Now connect everything ...

        # Spaces to in-Sockets.
        core.connect(self.state_space.with_batch_rank(), "states")
        core.connect(self.action_space.with_batch_rank(), "actions")
        core.connect(FloatBox(add_batch_rank=True), "rewards")
        core.connect(IntBox(2, add_batch_rank=True), "terminals")
        core.connect(BoolBox(), "deterministic")

        # States (from Env) into preprocessor -> into q-net
        core.connect("states", (self.preprocessor_stack, "input"))
        core.connect((self.preprocessor_stack, "output"), (self.policy, "input"), label="from_env")

        # Network output into Exploration's "nn_output" Socket -> into "act".
        core.connect((self.policy, "sample_deterministic"),
                     (self.exploration, "sample_deterministic"), label="from_env")
        core.connect((self.policy, "sample_stochastic"),
                     (self.exploration, "sample_stochastic"), label="from_env")
        core.connect((self.exploration, "action"), "act")

        # Actions, rewards, terminals into Merger.
        for in_ in ["actions", "rewards", "terminals"]:
            core.connect(in_, (self.merger, "/"+in_))
        # Preprocessed states into Merger.
        core.connect((self.preprocessor_stack, "output"), (self.merger, "/states"))
        # Merger's "output" into Memory's "records".
        core.connect((self.merger, "output"), (self.memory, "records"))
        # Memory's "add_records" functionality.
        core.connect((self.memory, "insert"), "add_records")

        # Memory's "sample" (to Splitter) and "num_records" (constant batch-size value).
        core.connect((self.memory, "sample"), (self.splitter, "input"))
        core.connect(self.update_spec["batch_size"], (self.memory, "num_records"))

        # Splitter's outputs.
        core.connect((self.splitter, "/states"), (self.policy, "input"), label="from_memory")
        core.connect((self.splitter, "/actions"), (self.loss_function, "actions"))
        core.connect((self.splitter, "/rewards"), (self.loss_function, "rewards"))
        core.connect((self.splitter, "/next_states"), (self.target_policy, "input"))

        # Loss-function needs both q-values (qnet and target).
        core.connect((self.policy, "nn_output"), (self.loss_function, "q_values"), label="from_memory")
        core.connect((self.target_policy, "nn_output"), (self.loss_function, "q_values_s_"))

        # Connect the Optimizer.
        core.connect((self.loss_function, "loss"), (self.optimizer, "loss"))
        core.connect((self.optimizer, "step"), "learn")

        # Add syncing capability for target-net.
        core.connect((self.policy, "sync_out"), (self.target_policy, "sync_in"))
        core.connect((self.target_policy, "sync_in"), "sync_target_qnet")

    def get_action(self, states, deterministic=False):
        self.timesteps += 1
        return self.graph_executor.execute("act", inputs=dict(state=states))

    def _observe_graph(self, states, actions, internals, reward, terminal):
        self.graph_executor.execute("add_records", inputs={
            self.input_names[0]: states,
            self.input_names[1]: actions,
            self.input_names[2]: reward,
            self.input_names[3]: terminal
        })

        # Every n steps, do a learn.
        if self.update_spec["unit"] == "timesteps" and (self.timesteps % self.update_spec["frequency"]) == 0:
            self.update()

    def update(self):
        return self.graph_executor.execute("learn")

    def __repr__(self):
        return "DQNAgent(doubleQ={})".format(self.double_q)
