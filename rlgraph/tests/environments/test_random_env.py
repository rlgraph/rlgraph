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

from rlgraph.spaces import IntBox, FloatBox
from rlgraph.environments import RandomEnv
from rlgraph.tests.test_util import recursive_assert_almost_equal


class TestRandomEnv(unittest.TestCase):
    """
    Tests creation, resetting and stepping through a deterministic RandomEnv.
    """
    def test_random_env(self):
        """
        Tests deterministic functionality of RandomEnv.
        """
        env = RandomEnv(state_space=FloatBox(shape=(2,2)), action_space=IntBox(2), deterministic=True)

        # Simple test runs with fixed actions.
        s = env.reset()
        recursive_assert_almost_equal(s, np.array([[0.77132064, 0.02075195], [0.63364823, 0.74880388]]))
        s, r, t, _ = env.step(env.action_space.sample())
        recursive_assert_almost_equal(s, np.array([[0.76053071, 0.16911084], [0.08833981, 0.68535982]]))
        s, r, t, _ = env.step(env.action_space.sample())
        recursive_assert_almost_equal(r, np.array(0.9177741225129434))
        s, r, t, _ = env.step(env.action_space.sample())
        self.assertEqual(t, False)
        s, r, t, _ = env.step(env.action_space.sample())
        recursive_assert_almost_equal(s, np.array([[0.65039718, 0.60103895], [0.8052232 , 0.52164715]]))
        s, r, t, _ = env.step(env.action_space.sample())
