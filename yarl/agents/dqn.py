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
from yarl.components.memories import Memory
from yarl.components.action_heads import ActionHead
from yarl.components.loss_functions import DQNLossFunction


class DQN(Agent):
    def __init__(self, memory_spec=None, **kwargs):
        """

        Args:
            memory_spec (Optional[dict,Memory]): The spec for the Memory to use for the DQN algorithm.
        """
        super(DQN, self).__init__(**kwargs)
        # Our memory component.
        self.memory = Memory.from_spec(memory_spec)
        # The target network.
        self.target_net = None

    def build_graph(self, core):
        # Define our interface.
        core.define_inputs("state", "action", "reward")
        core.define_outputs("act", "add_records", "reset_memory", "learn", "sync_target_qnet")

        # Add the Q-net, copy it (target-net) and add the target-net.
        core.add_component(self.neural_network)
        self.target_net = self.neural_network.copy(scope="target-net")
        core.add_component(self.target_net)

        # Add our Memory Component.
        core.add_component(self.memory)

        # Add an ActionHead to the q-net.
        core.add_component(ActionHead(action_space=self.action_space, epsilon_spec=self.exploration_spec))

        # Add the loss function and optimizer.
        core.add_component(DQNLossFunction())
        core.add_component(self.optimizer)


