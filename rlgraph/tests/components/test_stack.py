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
import numpy as np

from rlgraph.components.layers import Stack
from rlgraph.tests.dummy_components import Dummy1To1, Dummy2To1, Dummy1To2
from rlgraph.spaces import FloatBox, IntBox
from rlgraph.tests import ComponentTest


class TestStack(unittest.TestCase):
    """
    Tests for Stack Components of different length (number of sub-Components).
    """
    def test_one_sub_component(self):
        stack = Stack(Dummy1To1(constant_value=3.0), api_methods={"run"})
        test = ComponentTest(component=stack, input_spaces=dict(run=FloatBox()))

        test.test(("run", 4.6), expected_outputs=7.6)

    def test_two_sub_components(self):
        stack = Stack(Dummy1To1(scope="A", constant_value=3.0),
                      Dummy1To1(scope="B", constant_value=1.0),
                      api_methods={"run"})
        test = ComponentTest(component=stack, input_spaces=dict(run=FloatBox()))

        test.test(("run", 4.6), expected_outputs=np.array(8.6, dtype=np.float32))

    def test_two_sub_components_1to2_2to1(self):
        stack = Stack(Dummy1To2(scope="A", constant_value=1.5),
                      Dummy2To1(scope="B"),
                      api_methods={"run"})
        test = ComponentTest(component=stack, input_spaces=dict(run=FloatBox()))

        # Expect: (in + 1.5, in * 1.5) -> (1.5, 0.0) -> (1.5 + 0.0) = 1.5
        test.test(("run", 0.0), expected_outputs=np.array(1.5, dtype=np.float32))

    def test_two_sub_components_2to1_1to2(self):
        stack = Stack(Dummy2To1(scope="A"),
                      Dummy1To2(scope="B", constant_value=2.3),
                      api_methods={"run"})
        test = ComponentTest(component=stack, input_spaces=dict(run=(FloatBox(), float)))

        # Expect: (in1 + in2) -> 0.1 -> (0.1 + 2.3, 0.1 * 2.3)
        test.test(("run", [0.0, 0.1]), expected_outputs=(2.4, 0.23))
