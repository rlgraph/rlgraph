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

import unittest

from rlgraph.environments import GridWorld


class TestGridWorld(unittest.TestCase):
    """
    Tests creation, resetting and stepping through a deterministic GridWorld.
    """
    def test_2x2_grid_world(self):
        """
        Tests a minimalistic 2x2 GridWorld.
        """
        env = GridWorld(world="2x2")

        # Make everything deterministic.
        env.seed(55)

        # Simple test runs with fixed actions.
        # X=player's position
        s = env.reset()  # ["XH", " G"]  X=player's position
        self.assertTrue(s == 0)
        s, r, t, _ = env.step(2)  # down: [" H", "XG"]
        self.assertTrue(s == 1)
        self.assertTrue(r == -1.0)
        self.assertTrue(t is False)
        s, r, t, _ = env.step(1)  # right: [" H", " X"]
        self.assertTrue(s == 3)
        self.assertTrue(r == 1.0)
        self.assertTrue(t is True)

        env.reset()  # ["XH", " G"]  X=player's position
        s, r, t, _ = env.step(1)  # right: [" X", " G"] -> in the hole
        self.assertTrue(s == 2)
        self.assertTrue(r == -5.0)
        self.assertTrue(t is True)

        # Run against a wall.
        env.reset()  # ["XH", " G"]  X=player's position
        s, r, t, _ = env.step(3)  # left: ["XH", " G"]
        self.assertTrue(s == 0)
        self.assertTrue(r == -1.0)
        self.assertTrue(t is False)
        s, r, t, _ = env.step(2)  # down: [" H", "XG"]
        self.assertTrue(s == 1)
        self.assertTrue(r == -1.0)
        self.assertTrue(t is False)
        s, r, t, _ = env.step(0)  # up: ["XH", " G"]
        self.assertTrue(s == 0)
        self.assertTrue(r == -1.0)
        self.assertTrue(t is False)
