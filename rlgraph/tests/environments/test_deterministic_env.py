# Copyright 2018/2019 The RLgraph Authors, All Rights Reserved.
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
import unittest

from rlgraph.spaces import IntBox, FloatBox
from rlgraph.environments.deterministic_env import DeterministicEnv
from rlgraph.tests.test_util import recursive_assert_almost_equal


class TestDeterministicEnv(unittest.TestCase):
    """
    Tests creation, resetting and stepping through a DeterministicEnv.
    """
    def test_deterministic_env(self):
        """
        Tests functionality of DeterministicEnv.
        """
        env = DeterministicEnv(state_start=0.0, reward_start=50.0, steps_to_terminal=5)

        # Simple test runs with random actions.
        s = env.reset()
        recursive_assert_almost_equal(s, [0.0])

        # Perform 5 steps
        for i in range(5):
            s, r, t, _ = env.step(env.action_space.sample())
            recursive_assert_almost_equal(s, [1.0 + i])
            recursive_assert_almost_equal(r, 50.0 + i)
            if i == 4:
                self.assertTrue(t)
            else:
                self.assertFalse(t)

        s = env.reset()
        recursive_assert_almost_equal(s, [0.0])

        # Perform another 5 steps.
        for i in range(5):
            s, r, t, _ = env.step(env.action_space.sample())
            recursive_assert_almost_equal(s, [1.0 + i])
            recursive_assert_almost_equal(r, 50.0 + i)
            if i == 4:
                self.assertTrue(t)
            else:
                self.assertFalse(t)

