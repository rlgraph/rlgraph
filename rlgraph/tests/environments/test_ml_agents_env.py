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

import unittest

import numpy as np


class TestMLAgentsEnv(unittest.TestCase):
    """
    Tests creation, resetting and stepping through an openAI Atari Env.
    """
    def test_ml_agents_env(self):
        try:
            from rlgraph.environments import MLAgentsEnv
        except ImportError:
            print("MLAgents not installed -> skipping this test case.")
            return

        env = MLAgentsEnv()

        # Simple test runs with fixed actions.
        env.reset()
        for _ in range(100):
            actions = [env.action_space.sample() for _ in range(env.num_environments)]
            s, r, t, _ = env.step(actions)
            assert all(isinstance(r_, np.ndarray) for r_ in r)
            assert all(r_.dtype == np.float32 for r_ in r)
            assert all(isinstance(t_, bool) for t_ in t)

        env.terminate()
