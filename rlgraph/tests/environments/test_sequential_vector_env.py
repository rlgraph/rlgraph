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

from rlgraph.environments import SequentialVectorEnv
from rlgraph.tests.test_util import recursive_assert_almost_equal


class TestSequentialVectorEnv(unittest.TestCase):
    """
    Tests creation, resetting and stepping through a sequential vectorized Env with GridWorld entities.
    """
    def test_sequential_vector_env(self):
        num_envs = 4
        env = SequentialVectorEnv(num_environments=num_envs, env_spec={"type": "gridworld", "world": "2x2"})

        # Simple test runs with fixed actions.
        # X=player's position
        s = env.reset(index=0)  # ["XH", " G"]  X=player's position
        self.assertTrue(s == 0)

        s = env.reset_all()
        all(self.assertTrue(s_ == 0) for s_ in s)

        s, r, t, _ = env.step([2 for _ in range(num_envs)])  # down: [" H", "XG"]
        all(self.assertTrue(s_ == 1) for s_ in s)
        all(recursive_assert_almost_equal(r_, -0.1) for r_ in r)
        all(self.assertTrue(not t_) for t_ in t)

        s, r, t, _ = env.step([1 for _ in range(num_envs)])  # right: [" H", " X"]
        all(self.assertTrue(s_ == 3) for s_ in s)
        all(recursive_assert_almost_equal(r_, 1.0) for r_ in r)
        all(self.assertTrue(t_) for t_ in t)

        [env.reset(index=i) for i in range(num_envs)]  # ["XH", " G"]  X=player's position
        s, r, t, _ = env.step([1 for _ in range(num_envs)])  # right: [" X", " G"] -> in the hole
        all(self.assertTrue(s_ == 2) for s_ in s)
        all(self.assertTrue(r_ == -5.0) for r_ in r)
        all(self.assertTrue(t_) for t_ in t)

        # Run against a wall.
        env.reset_all()  # ["XH", " G"]  X=player's position
        s, r, t, _ = env.step([3 for _ in range(num_envs)])  # left: ["XH", " G"]
        all(self.assertTrue(s_ == 0) for s_ in s)
        all(recursive_assert_almost_equal(r_, -0.1) for r_ in r)
        all(self.assertTrue(not t_) for t_ in t)
        s, r, t, _ = env.step([2 for _ in range(num_envs)])  # down: [" H", "XG"]
        all(self.assertTrue(s_ == 1) for s_ in s)
        all(recursive_assert_almost_equal(r_, -0.1) for r_ in r)
        all(self.assertTrue(not t_) for t_ in t)
        s, r, t, _ = env.step([0 for _ in range(num_envs)])  # up: ["XH", " G"]
        all(self.assertTrue(s_ == 0) for s_ in s)
        all(recursive_assert_almost_equal(r_, -0.1) for r_ in r)
        all(self.assertTrue(not t_) for t_ in t)

