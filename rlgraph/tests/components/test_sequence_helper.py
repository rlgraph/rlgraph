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

from rlgraph.components.helpers.sequence_helper import SequenceHelper
from rlgraph.spaces import IntBox, FloatBox
from rlgraph.tests import ComponentTest, recursive_assert_almost_equal


class TestSequenceHelper(unittest.TestCase):
    input_spaces = dict(
        sequence_indices=IntBox(add_batch_rank=True),
        values=FloatBox(add_batch_rank=True),
        rewards=FloatBox(add_batch_rank=True),
        decay=float
    )

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
