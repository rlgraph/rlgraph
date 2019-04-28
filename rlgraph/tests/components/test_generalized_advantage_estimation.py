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
from rlgraph.components.helpers import GeneralizedAdvantageEstimation
from rlgraph.spaces import *
from rlgraph.tests import ComponentTest, recursive_assert_almost_equal


class TestGeneralizedAdvantageEstimation(unittest.TestCase):

    gamma = 0.99
    gae_lambda = 1.0

    rewards = FloatBox(add_batch_rank=True)
    baseline_values = FloatBox(add_batch_rank=True)
    terminals = BoolBox(add_batch_rank=True)
    sequence_indices = BoolBox(add_batch_rank=True)

    input_spaces = dict(
        rewards=rewards,
        baseline_values=baseline_values,
        terminals=terminals,
        sequence_indices=sequence_indices
    )

    @staticmethod
    def discount(x, gamma):
        # Discounts a single sequence.
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

    @staticmethod
    def discount_all(values, decay, terminal):
        # Discounts multiple sub-sequences by keeping track of terminals.
        discounted = []
        i = len(values) - 1
        prev_v = 0.0
        for v in reversed(values):
            # Arrived at new sequence, start over.
            if np.all(terminal[i]):
                print("Resetting discount after processing i = ", i)
                prev_v = 0.0

            # Accumulate prior value.
            accum_v = v + decay * prev_v
            discounted.append(accum_v)
            prev_v = accum_v

            i -= 1
        return list(reversed(discounted))

    def gae_helper(self, baseline, reward, gamma, gae_lambda, terminals, sequence_indices):
        # Bootstrap adjust.
        deltas = []
        start_index = 0
        i = 0
        sequence_indices[-1] = True
        for _ in range(len(baseline)):
            if np.all(sequence_indices[i]):
                # Compute deltas for this subsequence.
                # Cannot do this all at once because we would need the correct offsets for each sub-sequence.
                baseline_slice = list(baseline[start_index:i + 1])

                if np.all(terminals[i]):
                    print("Appending boot-strap val 0 at index.", i)
                    baseline_slice.append(0)
                else:
                    print("Appending boot-strap val {} at index {}.".format(baseline[i], i+1))
                    baseline_slice.append(baseline[i])
                adjusted_v = np.asarray(baseline_slice)

                # +1 because we want to include i-th value.
                delta = reward[start_index:i + 1] + gamma * adjusted_v[1:] - adjusted_v[:-1]
                print("Length for sequence deltas: ", len(delta))
                deltas.extend(delta)
                start_index = i + 1
            i += 1

        deltas = np.asarray(deltas)
        print("len deltas = ", len(deltas))
        return np.asarray(self.discount_all(deltas, gamma * gae_lambda, terminals))

    def test_with_manual_numbers_and_lambda_0_5(self):
        lambda_ = 0.5
        lg = lambda_ * self.gamma
        gae = GeneralizedAdvantageEstimation(gae_lambda=lambda_, discount=self.gamma)

        test = ComponentTest(component=gae, input_spaces=self.input_spaces)

        # Batch of 2 sequences.
        rewards_ = np.array([0.1, 0.2, 0.3])
        baseline_values_ = np.array([1.0, 2.0, 3.0])
        terminals_ = np.array([False, False, False])

        # Final sequence index must always be true.
        sequence_indices = np.array([False, False, True])
        input_ = [baseline_values_, rewards_, terminals_, sequence_indices]

        # Test TD-error outputs.
        td = np.array([1.08, 1.17, 0.27])
        test.test(("calc_td_errors", input_), expected_outputs=td, decimals=5)

        expected_gaes_manual = np.array([
            td[0] + lg * td[1] + lg * lg * td[2],
            td[1] + lg * td[2],
            td[2]
        ])
        expected_gaes_helper = self.gae_helper(
            baseline_values_, rewards_, self.gamma, lambda_, terminals_, sequence_indices
        )
        recursive_assert_almost_equal(expected_gaes_manual, expected_gaes_helper, decimals=5)
        advantages = test.test(("calc_gae_values", input_), expected_outputs=expected_gaes_manual)

        print("Rewards:", rewards_)
        print("Baseline-values:", baseline_values_)
        print("Terminals:", terminals_)
        print("Expected advantage:", expected_gaes_manual)
        print("Got advantage:", advantages)

        test.terminate()

    def test_single_non_terminal_sequence(self):
        gae = GeneralizedAdvantageEstimation(gae_lambda=self.gae_lambda, discount=self.gamma)

        test = ComponentTest(component=gae, input_spaces=self.input_spaces)

        rewards_ = self.rewards.sample(10)  #, fill_value=0.5)
        baseline_values_ = self.baseline_values.sample(10)  #, fill_value=1.0)
        terminals_ = self.terminals.sample(size=10, fill_value=False)

        # Final sequence index must always be true.
        sequence_indices = [False] * 10
        # Assume sequence indices = terminals here.
        input_ = [baseline_values_, rewards_, terminals_, sequence_indices]

        advantage_expected = self.gae_helper(
            baseline=baseline_values_,
            reward=rewards_,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            terminals=terminals_,
            sequence_indices=sequence_indices
        )

        advantage = test.test(("calc_gae_values", input_))
        recursive_assert_almost_equal(advantage_expected, advantage, decimals=5)
        print("Rewards:", rewards_)
        print("Baseline-values:", baseline_values_)
        print("Terminals:", terminals_)
        print("Expected advantage:", advantage_expected)
        print("Got advantage:", advantage)

        test.terminate()

    def test_multiple_sequences(self):
        gae = GeneralizedAdvantageEstimation(gae_lambda=self.gae_lambda, discount=self.gamma)

        test = ComponentTest(component=gae, input_spaces=self.input_spaces)

        rewards_ = self.rewards.sample(10)  #, fill_value=0.5)
        baseline_values_ = self.baseline_values.sample(10)  #, fill_value=1.0)
        terminals_ = [False] * 10
        terminals_[5] = True
        sequence_indices = [False] * 10
        sequence_indices[5] = True
        terminals_ = np.asarray(terminals_)

        input_ = [baseline_values_, rewards_, terminals_, sequence_indices]
        advantage_expected = self.gae_helper(
            baseline=baseline_values_,
            reward=rewards_,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            terminals=terminals_,
            sequence_indices=sequence_indices
        )

        print("Advantage expected:", advantage_expected)
        advantage = test.test(("calc_gae_values", input_))
        print("Got advantage = ", advantage)
        recursive_assert_almost_equal(advantage_expected, advantage, decimals=5)

        test.terminate()
