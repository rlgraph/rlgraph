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

from rlgraph.components.helpers import GeneralizedAdvantageEstimation
from rlgraph.spaces import *
from rlgraph.tests import ComponentTest
import scipy.signal


class TestGeneralizedAdvantageEstimation(unittest.TestCase):

    @staticmethod
    def discount(x, gamma):
        return scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]

    def gae_helper(self, baseline, reward, gamma, gae_lambda, terminated):
        # This is John Schulman's original way of implementing GAE.
        terminal_corrected_baseline = np.append(baseline, 0 if terminated else baseline[-1])
        deltas = reward + gamma * terminal_corrected_baseline[1:] - terminal_corrected_baseline[:-1]
        return self.discount(deltas, gamma * gae_lambda)

    def test_gae(self):
        gae = GeneralizedAdvantageEstimation(batch_size=10, gae_lambda=1.0, discount=0.99)

        rewards = FloatBox(add_batch_rank=True)
        baseline_values = FloatBox(add_batch_rank=True)
        terminals = IntBox(low=0, high=1, add_batch_rank=True)

        input_spaces = dict(
            rewards=rewards,
            baseline_values=baseline_values,
            terminals=terminals
        )
        test = ComponentTest(component=gae, input_spaces=input_spaces)

        rewards_ = rewards.sample(10)
        baseline_values_ = baseline_values.sample(10)
        terminals_ = terminals.sample(size=10, fill_value=0)
        input_ = [rewards_, baseline_values_, terminals_]

        advantage_expected = self.gae_helper(
            baseline=baseline_values_,
            reward=rewards_,
            gamma=0.99,
            gae_lambda=1.0,
            terminated=False
        )
        print(len(advantage_expected))
        print(advantage_expected)
        # print(test.test(("calc_gae_values", input_)))
