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
from rlgraph.components.helpers.sequence_helper import SequenceHelper
from rlgraph.spaces import FloatBox, BoolBox
from rlgraph.tests import ComponentTest, recursive_assert_almost_equal


class TestSequenceHelper(unittest.TestCase):

    input_spaces = dict(
        sequence_indices=BoolBox(add_batch_rank=True),
        terminals=BoolBox(add_batch_rank=True),
        values=FloatBox(add_batch_rank=True),
        rewards=FloatBox(add_batch_rank=True),
        decay=float
    )

    @staticmethod
    def decay_td_sequence(td_errors, decay=0.99, value_next=0.0):
        discounted_td_errors = np.zeros_like(td_errors)
        running_add = value_next
        for t in reversed(range(0, td_errors.size)):
            running_add = running_add * decay + td_errors[t]
            discounted_td_errors[t] = running_add
        return discounted_td_errors

    @staticmethod
    def deltas(baseline, reward, discount, terminals, sequence_values):
        """
        Computes expected 1-step TD errors over a sequence of rewards, terminals, sequence-indices:

        delta = reward + discount * bootstrapped_values[1:] - bootstrapped_values[:-1]
        """
        deltas = []
        start_index = 0
        i = 0
        for _ in range(len(baseline)):
            if np.all(sequence_values[i]):
                # Compute deltas for this sub-sequence.
                # Cannot do this all at once because we would need the correct offsets for each sub-sequence.
                baseline_slice = list(baseline[start_index:i + 1])

                # Boot-strap: If also terminal, with 0, else with last value.
                if np.all(terminals[i]):
                    print("Appending boot-strap val 0 at index.", i)
                    baseline_slice.append(0)
                else:
                    print("Appending boot-strap val {} at index {}.".format(baseline[i], i))
                    baseline_slice.append(baseline[i])

                adjusted_v = np.asarray(baseline_slice)

                print("adjusted_v", adjusted_v)
                print("adjusted_v[1:]", adjusted_v[1:])
                print("adjusted_v[:-1]",  adjusted_v[:-1])

                # +1 because we want to include i-th value.
                delta = reward[start_index:i + 1] + discount * adjusted_v[1:] - adjusted_v[:-1]
                deltas.extend(delta)
                start_index = i + 1
            i += 1

        return np.array(deltas)

    def test_calc_sequence_lengths(self):
        """
        Tests counting sequence lengths based on terminal configurations.
        """
        sequence_helper = SequenceHelper()
        test = ComponentTest(component=sequence_helper, input_spaces=self.input_spaces)
        input_ = np.asarray([0, 0, 0, 0])
        test.test(("calc_sequence_lengths", input_), expected_outputs=[4])

        input_ = np.asarray([0, 0, 1, 0])
        test.test(("calc_sequence_lengths", input_), expected_outputs=[3, 1])

        input_ = np.asarray([1, 1, 1, 1])
        test.test(("calc_sequence_lengths", input_), expected_outputs=[1, 1, 1, 1])

        input_ = np.asarray([1, 0, 0, 1])
        test.test(("calc_sequence_lengths", input_), expected_outputs=[1, 3])

    def test_bootstrapping(self):
        """
        Tests boot-strapping for GAE purposes.
        """
        sequence_helper = SequenceHelper()
        discount = 0.99

        test = ComponentTest(component=sequence_helper, input_spaces=self.input_spaces)

        # No terminals - just boot-strap with final sequence index.
        values = np.asarray([1.0, 2.0, 3.0, 4.0])
        rewards = np.asarray([0, 0, 0, 0])
        sequence_indices = np.asarray([0, 0, 0, 1])
        terminals = np.asarray([0, 0, 0, 0])

        expected_deltas = self.deltas(values, rewards, discount, terminals, sequence_indices)
        deltas = test.test(("bootstrap_values", [rewards, values, terminals, sequence_indices]))
        recursive_assert_almost_equal(expected_deltas, deltas, decimals=5)

        # Final index is also terminal.
        values = np.asarray([1.0, 2.0, 3.0, 4.0])
        rewards = np.asarray([0, 0, 0, 0])
        sequence_indices = np.asarray([0, 0, 0, 1])
        terminals = np.asarray([0, 0, 0, 1])

        expected_deltas = self.deltas(values, rewards, discount, terminals, sequence_indices)
        deltas = test.test(("bootstrap_values", [rewards, values, terminals, sequence_indices]))
        recursive_assert_almost_equal(expected_deltas, deltas, decimals=5)

        # Mixed: i = 1 is also terminal, i = 3 is only sequence.
        values = np.asarray([1.0, 2.0, 3.0, 4.0])
        rewards = np.asarray([0, 0, 0, 0])
        sequence_indices = np.asarray([0, 1, 0, 1])
        terminals = np.asarray([0, 1, 0, 0])

        expected_deltas = self.deltas(values, rewards, discount, terminals, sequence_indices)
        deltas = test.test(("bootstrap_values", [rewards, values, terminals, sequence_indices]))
        recursive_assert_almost_equal(expected_deltas, deltas, decimals=5)

    def test_calc_decays(self):
        """
        Tests counting sequence lengths based on terminal configurations.
        """
        sequence_helper = SequenceHelper()
        decay_value = 0.5

        test = ComponentTest(component=sequence_helper, input_spaces=self.input_spaces)
        input_ = np.asarray([0, 0, 0, 0])
        expected_decays = [1.0, 0.5, 0.25, 0.125]
        lengths, decays = test.test(("calc_sequence_decays", [input_, decay_value]))

        # Check lengths and decays.
        recursive_assert_almost_equal(x=lengths, y=[4])
        recursive_assert_almost_equal(x=decays, y=expected_decays)

        input_ = np.asarray([0, 0, 1, 0])
        expected_decays = [1.0, 0.5, 0.25, 1.0]
        lengths, decays = test.test(("calc_sequence_decays", [input_, decay_value]))

        recursive_assert_almost_equal(x=lengths, y=[3, 1])
        recursive_assert_almost_equal(x=decays, y=expected_decays)

        input_ = np.asarray([1, 1, 1, 1])
        expected_decays = [1.0, 1.0, 1.0, 1.0]
        lengths, decays = test.test(("calc_sequence_decays", [input_, decay_value]))

        recursive_assert_almost_equal(x=lengths, y=[1, 1, 1, 1])
        recursive_assert_almost_equal(x=decays, y=expected_decays)

    def test_reverse_apply_decays_to_sequence(self):
        """
        Tests reverse decaying a sequence of 1-step TD errors for GAE.
        """
        sequence_helper = SequenceHelper()
        decay_value = 0.5

        test = ComponentTest(component=sequence_helper, input_spaces=self.input_spaces)
        td_errors = np.asarray([0.1, 0.2, 0.3, 0.4])
        indices = np.array([0, 0, 0, 1])
        expected_output_sequence_manual = np.asarray([
            0.1 + 0.5 * 0.2 + 0.25 * 0.3 + 0.125 * 0.4,
            0.2 + 0.5 * 0.3 + 0.25 * 0.4,
            0.3 + 0.5 * 0.4,
            0.4
        ])
        expected_output_sequence_numpy = self.decay_td_sequence(td_errors, decay=decay_value)
        recursive_assert_almost_equal(expected_output_sequence_manual, expected_output_sequence_numpy)
        test.test(
            ("reverse_apply_decays_to_sequence", [td_errors, indices, decay_value]),
            expected_outputs=expected_output_sequence_manual
        )
