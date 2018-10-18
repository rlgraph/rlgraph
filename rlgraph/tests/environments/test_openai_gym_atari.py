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
import unittest

from rlgraph.environments import OpenAIGymEnv


class TestOpenAIAtariEnv(unittest.TestCase):
    """
    Tests creation, resetting and stepping through an openAI Atari Env.
    """
    def test_openai_atari_env(self):
        env = OpenAIGymEnv("Pong-v0")

        # Simple test runs with fixed actions.
        s = env.reset()
        # Assert we have pixels.
        self.assertGreaterEqual(np.mean(s), 0)
        self.assertLessEqual(np.mean(s), 255)
        accum_reward = 0.0
        for _ in range(100):
            s, r, t, _ = env.step(env.action_space.sample())
            assert isinstance(r, np.ndarray)
            assert r.dtype == np.float32
            assert isinstance(t, bool)
            self.assertGreaterEqual(np.mean(s), 0)
            self.assertLessEqual(np.mean(s), 255)
            accum_reward += r

        print("Accumulated Reward: ".format(accum_reward))
