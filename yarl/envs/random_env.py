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

import math
import numpy as np
import random
from six.moves import xrange
import time

from yarl.envs import Env
import yarl.spaces as spaces


class RandomEnv(Env):
    """
    An Env producing random states no matter what actions come in.
    """
    def __init__(self, state_space, action_space, reward_space=None, terminal_prob=0.1):
        """
        Args:
            reward_space (Union[dict,Space]): The reward Space from which to randomly sample for each step.
            terminal_prob (Union[dict,Space]): The probability with which an episode ends for each step.
        """
        super(RandomEnv, self).__init__(state_space=state_space, action_space=action_space)

        self.reward_space = spaces.Space.from_spec(reward_space) or spaces.FloatBox(-1.0, 1.0)
        self.terminal_prob = terminal_prob

    def seed(self, seed=None):
        if seed is None:
            seed = time.time()
        np.random.seed(seed)
        return seed

    def reset(self):
        return self.step()

    def step(self):
        return self.state_space.sample(), self.reward_space.sample(),\
               np.random.choice([True, False], p=[self.terminal_prob, 1.0 - self.terminal_prob]), None

    def __str__(self):
        return "RandomEnv()"
