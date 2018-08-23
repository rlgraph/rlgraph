# Copyright 2018 The Rlgraph Authors, All Rights Reserved.
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

from rlgraph.components.common.slice import Slice
from rlgraph.spaces import FloatBox, IntBox
from rlgraph.tests.component_test import ComponentTest


class TestSlice(unittest.TestCase):

    def test_slice_with_squeeze(self):
        slicer = Slice(squeeze=True)
        input_space = FloatBox(shape=(2, 2, 3), add_batch_rank=True, add_time_rank=True, time_major=True)
        test = ComponentTest(component=slicer, input_spaces=dict(
            preprocessing_inputs=input_space,
            start_index=IntBox(),
            end_index=IntBox()
        ))

        # Time-steps=3, Batch=5
        inputs = input_space.sample(size=(3, 5))
        expected = inputs[1]
        test.test(("slice", [inputs, 1, 2]), expected_outputs=expected)

        expected = inputs[0:2]
        test.test(("slice", [inputs, 0, 2]), expected_outputs=expected)

        expected = inputs[0]
        test.test(("slice", [inputs, 0, 1]), expected_outputs=expected)

    def test_slice_without_squeeze(self):
        slicer = Slice(squeeze=False)
        input_space = FloatBox(shape=(1, 4, 5), add_batch_rank=True)
        test = ComponentTest(component=slicer, input_spaces=dict(
            preprocessing_inputs=input_space,
            start_index=IntBox(),
            end_index=IntBox()
        ))

        # Time-steps=3, Batch=5
        inputs = input_space.sample(size=4)
        expected = np.asarray([inputs[1]])  # Add the not-squeezed rank back to expected.
        test.test(("slice", [inputs, 1, 2]), expected_outputs=expected)

        expected = inputs[0:2]
        test.test(("slice", [inputs, 0, 2]), expected_outputs=expected)

        expected = np.asarray([inputs[0]])
        test.test(("slice", [inputs, 0, 1]), expected_outputs=expected)
