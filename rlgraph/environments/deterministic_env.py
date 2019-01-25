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

import numpy as np

from rlgraph.environments import Environment
import rlgraph.spaces as spaces


class DeterministicEnv(Environment):
    """
    An Env producing a simple float state starting from `state_start` after reset and counting upwards in steps of
    1.0 (regardless of the actions). Same goes for the reward signal, which starts from `reward_start`.
    The action space is IntBox(2). Episodes terminate after always `steps_to_terminal` steps.
    """
    def __init__(self, state_start=0.0, reward_start=-100.0, steps_to_terminal=10):
        """
        Args:
            state_start (float): State to start with after reset.
            reward_start (float): Reward to start with (after first action) after a reset.
            steps_to_terminal (int): Number of steps after which a terminal signal is raised.
        """
        super(DeterministicEnv, self).__init__(state_space=spaces.FloatBox(), action_space=spaces.IntBox(2))

        self.state_start = state_start
        self.reward_start = reward_start
        self.steps_to_terminal = steps_to_terminal

        self.state = state_start
        self.reward = reward_start
        self.steps_into_episode = 0

    def seed(self, seed=None):
        return seed

    def reset(self):
        self.steps_into_episode = 0
        self.state = self.state_start
        self.reward = self.reward_start
        return np.array([self.state], dtype=np.float32)

    def reset_flow(self):
        return self.reset()

    def step(self, actions=None):
        if actions is not None:
            assert self.action_space.contains(actions), \
                "ERROR: Given action ({}) in step is not part of action Space ({})!".format(actions, self.action_space)

        self.state += 1.0
        reward = self.reward
        self.reward += 1.0
        self.steps_into_episode += 1
        terminal = False
        if self.steps_into_episode >= self.steps_to_terminal:
            terminal = True
        return np.array([self.state], dtype=np.float32), np.array(reward, dtype=np.float32), terminal, None

    def step_flow(self, actions=None):
        ret = self.step(actions)
        state = ret[0]
        if ret[2]:
            state = self.reset()
        return state, ret[1], ret[2]

    def __str__(self):
        return "DeterministicEnv()"
