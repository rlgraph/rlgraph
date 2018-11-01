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


class TestGeneralizedAdvantageEstimation(unittest.TestCase):

    gamma = 0.99
    gae_lambda = 1.0

    @staticmethod
    def discount(x, gamma):
        discounted = []
        prev = 0
        index = 0
        # Apply discount to value.
        for val in reversed(x):
            decayed = prev + val * pow(gamma, index)
            discounted.append(decayed)
            index += 1
            prev = decayed
        return list(reversed(discounted))

    def gae_helper(self, baseline, reward, gamma, gae_lambda, terminated):
        # N.b. this only works for a subsequence
        # This is John Schulman's original way of implementing GAE.
        terminal_corrected_baseline = np.append(baseline, 0 if terminated else baseline[-1])
        deltas = reward + gamma * terminal_corrected_baseline[1:] - terminal_corrected_baseline[:-1]
        return self.discount(deltas, gamma * gae_lambda)

    def test_gae(self):
        gae = GeneralizedAdvantageEstimation(gae_lambda=self.gae_lambda, discount=self.gamma)

        rewards = FloatBox(add_batch_rank=True)
        baseline_values = FloatBox(add_batch_rank=True)
        terminals = IntBox(low=0, high=1, add_batch_rank=True)

        input_spaces = dict(
            rewards=rewards,
            baseline_values=baseline_values,
            terminals=terminals
        )
        test = ComponentTest(component=gae, input_spaces=input_spaces)

        rewards_ = rewards.sample(10, fill_value=0.5)
        baseline_values_ = baseline_values.sample(10, fill_value=1.0)
        terminals_ = terminals.sample(size=10, fill_value=0)
        input_ = [baseline_values_, rewards_, terminals_]

        advantage_expected, deltas_expected = self.gae_helper(
            baseline=baseline_values_,
            reward=rewards_,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            terminated=False
        )

        print("Advantage expected:", advantage_expected)
        advantage, deltas = test.test(("calc_gae_values", input_))
        print("Got advantage = ", advantage)
