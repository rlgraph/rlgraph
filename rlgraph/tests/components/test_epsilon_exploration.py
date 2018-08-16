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

from rlgraph.components.explorations.exploration import EpsilonExploration
from rlgraph.components.common.decay_components import LinearDecay
from rlgraph.spaces import *
from rlgraph.tests import ComponentTest
from rlgraph.tests.test_util import recursive_assert_almost_equal


class TestEpslionExploration(unittest.TestCase):

    def test_epsilon_exploration_at_single_time_steps(self):
        time_step_space = IntBox(10000)
        sample_space = FloatBox(add_batch_rank=True)

        np.random.seed(10)

        # The Component(s) to test.
        decay_component = LinearDecay(from_=1.0, to_=0.15, start_timestep=0, num_timesteps=10000)
        epsilon_component = EpsilonExploration(decay_spec=decay_component)
        test = ComponentTest(component=epsilon_component, input_spaces=dict(
            sample=sample_space, time_step=time_step_space
        ))

        # Take n uniform samples over the time space and then check,
        # whether we have a decent distribution of do_explore values.
        time_steps = time_step_space.sample(size=100)
        out = np.ndarray(shape=(100, 10), dtype=np.bool_)
        for i, time_step in enumerate(time_steps):
            # Each time step, get epsilon decisions for a batch of samples.
            sample = sample_space.sample(10)
            out[i, :] = test.test(("do_explore", [sample, time_step]))

        # As we are going from epsilon 1.0 to 0.1, assert that we are slightly smaller than 0.5.
        mean = out.mean()
        self.assertAlmostEqual(mean, 0.6, places=1)
        self.assertGreater(mean, 0.5)

        # Take n samples at the end (+10000) of the exploration (almost no exploration).
        time_steps = time_step_space.sample(size=100) + 10000
        out = np.ndarray(shape=(100, 10), dtype=np.bool_)
        for i, time_step in enumerate(time_steps):
            # Each time step, get epsilon decisions for a batch of samples.
            sample = sample_space.sample(10)
            out[i, :] = test.test(("do_explore", [sample, time_step]))

        # As we are going from epsilon 1.0 to 0.0, assert that we are slightly smaller than 0.5.
        mean = out.mean()
        self.assertAlmostEqual(mean, 0.15, places=1)

    """
    TAKE OUT FOR NOW: doesn't make sense to explore with a sequence of time-steps at the same time.
    Will revisit when we look into multi-agent RL.
    def test_epsilon_exploration_with_time_rank(self):
        time_step_space = IntBox(add_time_rank=True)
        sample_space = FloatBox(add_batch_rank=True, add_time_rank=True, time_major=True)

        # The Component(s) to test.
        decay_component = LinearDecay(from_=1.0, to_=0.0, start_timestep=0, num_timesteps=1000)
        epsilon_component = EpsilonExploration(decay_spec=decay_component)
        test = ComponentTest(component=epsilon_component, input_spaces=dict(
            sample=sample_space, time_step=time_step_space
        ))

        # Values to pass at once (one time x batch pass).
        time_steps = np.array([0, 1, 2, 25, 50, 100, 110, 112, 120, 130, 150, 180, 190, 195, 200, 201, 210, 250, 386,
                               670, 789, 900, 923, 465, 894, 91, 1000])

        # Only pass in sample (zeros) for the batch rank (5).
        out = test.test(("do_explore", [np.zeros(shape=(27, 5)), time_steps]), expected_outputs=None)
        recursive_assert_almost_equal(out[0], [True] * 5)
        recursive_assert_almost_equal(out[-1], [False] * 5)
        mean = out.mean()
        self.assertAlmostEqual(mean, 0.65, places=1)
    """
