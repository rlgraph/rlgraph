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

from yarl.agents import Agent
from yarl.components.common import Merger, Splitter
from yarl.components.memories import Memory
from yarl.components.action_heads import ActionHead
from yarl.components.loss_functions import DQNLossFunction
from yarl.spaces import Dict, IntBox


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
        self.double_q = double_q
        self.duelling_q = duelling_q

        # The target network (is synced from q-net every n steps).
        self.target_net = None
        # The global copy of the q-net (if we are running in distributed mode).
        self.global_qnet = None

        self.action_head = ActionHead(action_space=self.action_space, epsilon_spec=self.exploration_spec)
        self.record_space = Dict(states=self.state_space, actions=self.action_space, rewards=float, terminal=IntBox(1))
        self.input_names = ["state", "action", "reward", "terminal"]
        self.merger = Merger(output_space=self.record_space, input_names=self.input_names)
        self.splitter = Splitter(input_space=self.record_space)
        self.loss_function = DQNLossFunction(double_q=self.double_q)

    def build_graph(self, core):
        # Define our interface.
        core.define_inputs(*self.input_names)
        core.define_outputs("act", "add_records", "reset_memory", "learn", "sync_target_qnet")

        # Add the Q-net, copy it (target-net) and add the target-net.
        self.target_net = self.neural_network.copy(scope="target-net")
        core.add_component(self.neural_network, self.target_net)
        # Add an ActionHead for the q-net (target net doesn't need one).
        core.add_component(self.action_head)

        # If we are in distributed mode, add a global qnet as well.
        if self.execution_spec[""] == "distributed":
            self.global_qnet = self.neural_network.copy(scope="global-qnet", global_component=True)
            core.add_component(self.global_qnet)

        # Add our Memory Component plus merger and splitter.
        core.add_components(self.memory, self.merger, self.splitter)

        # Add the loss function and optimizer.
        core.add_components(self.loss_function, self.optimizer)

        # Now connect everything ...

        # States into preprocessor -> into qnet
        core.connect("state", (self.preprocessor_stack, "input"))
        core.connect((self.preprocessor_stack, "output"), (self.neural_network, "input"), label="from_env")

        # Network output into ActionHead's "nn_output" Socket -> into "act".
        core.connect((self.neural_network, "output"), (self.action_head, "nn_output"), label="from_env")
        core.connect((self.action_head, "nn_output"), "act")

        # Actions, rewards, terminals into Merger.
        for in_ in self.input_names:
            if in_ != "states":
                core.connect(in_, (self.merger, in_))
        # Preprocessed states into Merger.
        core.connect((self.preprocessor_stack, "output"), (self.merger, "states"))
        # Merger's "output" into Memory's "records".
        core.connect((self.merger, "output"), (self.memory, "records"))
        # Memory's "add_records" functionality.
        core.connect((self.memory, "add_records"), "add_records")

        # Memory's "sample" (to Splitter) and "num_records" (constant batch-size value).
        core.connect((self.memory, "sample"), (self.splitter, "input"))
        core.connect(self.update_spec["batch_size"], (self.memory, "num_records"))

        # Splitter's outputs.
        core.connect((self.splitter, "states"), (self.neural_network, "input"), label="from_memory")
        core.connect((self.splitter, "actions"), (self.loss_function, "actions"))
        core.connect((self.splitter, "rewards"), (self.loss_function, "rewards"))
        core.connect((self.splitter, "next_states"), (self.target_net, "input"))

        # Loss-function needs both q-values (qnet and target).
        core.connect((self.neural_network, "output"), (self.loss_function, "q_values"), label="from_memory")
        core.connect((self.target_net, "output"), (self.loss_function, "q_values_s_"))

        # Connect the Optimizer.
        core.connect((self.loss_function, "loss"), (self.optimizer, "loss"))
        core.connect((self.optimizer, "step"), "learn")

        # Add syncing capability for target-net.
        core.connect((self.neural_network, "synch_out"), (self.target_net, "synch_in"))
        core.connect((self.target_net, "synch_in"), "synch_target_qnet")

    def __repr__(self):
        return "dqn_agent"
