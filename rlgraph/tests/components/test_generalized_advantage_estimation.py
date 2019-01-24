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
        i = 0
        length = 0
        prev_v = 0.0
        for v in reversed(values):
            # Accumulate prior value.
            accum_v = prev_v + v * pow(decay, length)
            discounted.append(accum_v)
            prev_v = accum_v

            # Arrived at new sequence, start over.
            if np.all(terminal[i]):
                print("Resetting discount after processing i = ", i)
                length = 0
                prev_v = 0.0

            # Increase length of current sub-sequence.
            length += 1
            i += 1
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
                    print("Appending boot-strap val {} at index {}.".format(baseline[i], i))
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

    def test_single_non_terminal_sequence(self):
        gae = GeneralizedAdvantageEstimation(gae_lambda=self.gae_lambda, discount=self.gamma)

        test = ComponentTest(component=gae, input_spaces=self.input_spaces)

        rewards_ = self.rewards.sample(10, fill_value=0.5)
        baseline_values_ = self.baseline_values.sample(10, fill_value=1.0)
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
        print("Expected advantage:", advantage_expected)
        print("Got advantage:", advantage)

        test.terminate()

    def test_multiple_sequences(self):
        gae = GeneralizedAdvantageEstimation(gae_lambda=self.gae_lambda, discount=self.gamma)

        test = ComponentTest(component=gae, input_spaces=self.input_spaces)

        rewards_ = self.rewards.sample(10, fill_value=0.5)
        baseline_values_ = self.baseline_values.sample(10, fill_value=1.0)
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
