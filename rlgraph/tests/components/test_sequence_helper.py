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

from rlgraph.components.helpers.sequence_helper import SequenceHelper
from rlgraph.spaces import IntBox
from rlgraph.tests import ComponentTest
import numpy as np


class TestSequenceHelper(unittest.TestCase):

    def test_calc_sequence_lengths(self):
        """
        Tests counting sequence lengths based on terminal configurations.
        """
        sequence_helper = SequenceHelper()
        sequence_indices = dict(sequence_indices=IntBox(add_batch_rank=True))
        test = ComponentTest(component=sequence_helper, input_spaces=sequence_indices)
        input_ = np.asarray([0, 0, 0, 0])
        test.test(("calc_sequence_lengths", input_), expected_outputs=[4])

        input_ = np.asarray([0, 0, 1, 0])
        test.test(("calc_sequence_lengths", input_), expected_outputs=[3, 1])

        input_ = np.asarray([1, 1, 1, 1])
        test.test(("calc_sequence_lengths", input_), expected_outputs=[1, 1, 1, 1])

        input_ = np.asarray([1, 0, 0, 1])
        test.test(("calc_sequence_lengths", input_), expected_outputs=[1, 3])

