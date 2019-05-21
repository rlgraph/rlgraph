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

from rlgraph.environments import GridWorld
from rlgraph.tests.test_util import recursive_assert_almost_equal


class TestGridWorld(unittest.TestCase):
    """
    Tests creation, resetting and stepping through a deterministic GridWorld.
    """
    def test_2x2_grid_world(self):
        """
        Tests a minimalistic 2x2 GridWorld.
        """
        env = GridWorld(world="2x2")

        # Simple test runs with fixed actions.
        # X=player's position
        s = env.reset()  # ["XH", " G"]  X=player's position
        self.assertTrue(s == 0)
        s, r, t, _ = env.step(2)  # down: [" H", "XG"]
        self.assertTrue(s == 1)
        recursive_assert_almost_equal(r, -0.1)
        self.assertTrue(not t)
        s, r, t, _ = env.step(1)  # right: [" H", " X"]
        self.assertTrue(s == 3)
        recursive_assert_almost_equal(r, 1.0)
        self.assertTrue(t)

        env.reset()  # ["XH", " G"]  X=player's position
        s, r, t, _ = env.step(1)  # right: [" X", " G"] -> in the hole
        self.assertTrue(s == 2)
        self.assertTrue(r == -5.0)
        self.assertTrue(t)

        # Run against a wall.
        env.reset()  # ["XH", " G"]  X=player's position
        s, r, t, _ = env.step(3)  # left: ["XH", " G"]
        self.assertTrue(s == 0)
        recursive_assert_almost_equal(r, -0.1)
        self.assertTrue(not t)
        s, r, t, _ = env.step(2)  # down: [" H", "XG"]
        self.assertTrue(s == 1)
        recursive_assert_almost_equal(r, -0.1)
        self.assertTrue(not t)
        s, r, t, _ = env.step(0)  # up: ["XH", " G"]
        self.assertTrue(s == 0)
        recursive_assert_almost_equal(r, -0.1)
        self.assertTrue(not t)

    def test_2x2_grid_world_using_flow_methods(self):
        """
        Tests a minimalistic 2x2 GridWorld.
        """
        env = GridWorld(world="2x2")

        # Simple test runs with fixed actions.
        # X=player's position
        s, r, t = env.step_flow(2)  # down: [" H", "XG"]
        self.assertTrue(s == 1)
        recursive_assert_almost_equal(r, -0.1)
        self.assertTrue(not t)
        s, r, t = env.step_flow(1)  # right: [" H", " X"]
        self.assertTrue(s == 0)
        self.assertTrue(r == 1.0)
        self.assertTrue(t)

        s, r, t = env.step_flow(1)  # right: [" X", " G"] -> in the hole
        self.assertTrue(s == 0)
        self.assertTrue(r == -5.0)
        self.assertTrue(t)

        # Run against a wall.
        s, r, t = env.step_flow(3)  # left: ["XH", " G"]
        self.assertTrue(s == 0)
        recursive_assert_almost_equal(r, -0.1)
        self.assertTrue(not t)
        s, r, t = env.step_flow(2)  # down: [" H", "XG"]
        self.assertTrue(s == 1)
        recursive_assert_almost_equal(r, -0.1)
        self.assertTrue(not t)
        s, r, t = env.step_flow(0)  # up: ["XH", " G"]
        self.assertTrue(s == 0)
        recursive_assert_almost_equal(r, -0.1)
        self.assertTrue(not t)

    def test_4x4_grid_world_with_container_actions(self):
        """
        Tests a 4x4 GridWorld using forward+turn+jump container actions.
        """
        env = GridWorld(world="4x4", action_type="ftj", state_representation="xy+orientation")

        # Simple test runs with fixed actions.

        # Fall into hole.
        s = env.reset()  # [0, 0, 0] (x, y, orientation)
        recursive_assert_almost_equal(s, [0, 0, 0, 1])
        s, r, t, _ = env.step(dict(turn=2, forward=2))  # turn=2 (right), move=2 (forward), jump=0
        recursive_assert_almost_equal(s, [1, 0, 1, 0])
        recursive_assert_almost_equal(r, -0.1)
        self.assertTrue(not t)
        s, r, t, _ = env.step(dict(turn=2, forward=1))  # turn=2 (right), move=1 (stay), jump=0
        recursive_assert_almost_equal(s, [1, 0, 0, -1])
        recursive_assert_almost_equal(r, -0.1)
        self.assertTrue(not t)
        s, r, t, _ = env.step(dict(turn=1, forward=2))  # turn=1 (no turn), move=2 (forward), jump=0
        recursive_assert_almost_equal(s, [1, 1, 0, -1])
        self.assertTrue(r == -5.0)
        self.assertTrue(t)

        # Jump quite a lot and reach goal.
        env.reset()  # [0, 0, 0] (x, y, orientation)
        s, r, t, _ = env.step(dict(turn=2, forward=1))
        recursive_assert_almost_equal(s, [0, 0, 1, 0])
        recursive_assert_almost_equal(r, -0.1)
        self.assertTrue(not t)
        s, r, t, _ = env.step(dict(turn=1, forward=1, jump=1))
        recursive_assert_almost_equal(s, [2, 0, 1, 0])
        recursive_assert_almost_equal(r, -0.1)
        self.assertTrue(not t)
        s, r, t, _ = env.step(dict(turn=2, forward=2))
        recursive_assert_almost_equal(s, [2, 1, 0, -1])
        recursive_assert_almost_equal(r, -0.1)
        self.assertTrue(not t)
        s, r, t, _ = env.step(dict(turn=1, forward=2, jump=1))
        recursive_assert_almost_equal(s, [2, 3, 0, -1])
        recursive_assert_almost_equal(r, -0.1)
        self.assertTrue(not t)
        s, r, t, _ = env.step(dict(turn=2, forward=0))
        recursive_assert_almost_equal(s, [3, 3, -1, 0])
        self.assertTrue(r == 1.0)
        self.assertTrue(t)

        # Run against a wall.
        env.reset()  # [0, 0, 0] (x, y, orientation)
        s, r, t, _ = env.step(dict(turn=1, forward=0))
        recursive_assert_almost_equal(s, [0, 1, 0, 1])
        recursive_assert_almost_equal(r, -0.1)
        self.assertTrue(not t)
        s, r, t, _ = env.step(dict(turn=0, forward=2))
        recursive_assert_almost_equal(s, [0, 1, -1, 0])
        recursive_assert_almost_equal(r, -0.1)
        self.assertTrue(not t)

        # Jump over a hole (no reset).
        s, r, t, _ = env.step(dict(turn=2, forward=1))  # turn around
        s, r, t, _ = env.step(dict(turn=2, forward=1))
        recursive_assert_almost_equal(s, [0, 1, 1, 0])
        recursive_assert_almost_equal(r, -0.1)
        self.assertTrue(not t)
        s, r, t, _ = env.step(dict(turn=1, forward=1, jump=1))
        recursive_assert_almost_equal(s, [2, 1, 1, 0])
        recursive_assert_almost_equal(r, -0.1)
        self.assertTrue(not t)

