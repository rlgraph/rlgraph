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

import logging
import numpy as np
import unittest

from rlgraph.tests import ComponentTest
from rlgraph.utils import root_logger
from rlgraph.tests.dummy_components import *


class TestSingleComponents(unittest.TestCase):
    """
    Tests for different ways to place different, but single sub-Components into the core.
    """
    root_logger.setLevel(level=logging.INFO)

    def test_single_component_with_single_api_method(self):
        """
        'A' is 1to1: send "input" through A, receive output.
        """
        a = Dummy1To1(scope="A")
        test = ComponentTest(component=a, input_spaces=dict(input_=float))
        # Expected: in + 1.0
        test.test(("run", 1.1), expected_outputs=2.1)

    def test_1to1_component(self):
        """
        Adds a single component with 1-to-1 graph_fn to the core and passes a value through it.
        """
        component = Dummy1To1(scope="dummy")
        test = ComponentTest(component=component, input_spaces=dict(input_=float))

        # Expected output: input + 1.0
        test.test(("run", 1.0), expected_outputs=2.0)
        test.test(("run", -5.0), expected_outputs=-4.0)

    def test_2to1_component(self):
        """
        Adds a single component with 2-to-1 graph_fn to the core and passes 2 values through it.
        """
        component = Dummy2To1(scope="dummy")
        test = ComponentTest(component=component, input_spaces=dict(input1=float, input2=float))

        # Expected output: input1 + input2
        test.test(("run", [1.0, 2.9]), expected_outputs=3.9)
        test.test(("run", [4.9, -0.1]), expected_outputs=np.array(4.8, dtype=np.float32))

    def test_1to2_component(self):
        """
        Adds a single component with 1-to-2 graph_fn to the core and passes a value through it.
        """
        component = Dummy1To2(scope="dummy", constant_value=1.3)
        test = ComponentTest(component=component, input_spaces=dict(input_=float))

        # Expected outputs: (input, input+1.0)
        test.test(("run", 1.0), expected_outputs=[2.3, 1.3])
        test.test(("run", 4.6), expected_outputs=[5.9, 5.98], decimals=3)

    def test_0to1_component(self):
        """
        Adds a single component with 0-to-1 graph_fn to the core and passes a value through it.
        """
        component = Dummy0To1(scope="dummy", var_value=5.0)
        test = ComponentTest(component=component, input_spaces=None)

        # Expected outputs: `var_value` passed into ctor.
        test.test(("run", None), expected_outputs=5.0)

    def test_2to1_component_with_int_input_space(self):
        """
        Adds a single component with 1-to-1 graph_fn to the core and blocks the input with a constant value.
        """
        component = Dummy2To1(scope="dummy")
        test = ComponentTest(component=component, input_spaces=dict(input1=int, input2=int))

        # Expected output: in1 + in2
        test.test(("run", [5, 4]), expected_outputs=9)

    def test_2to1_component_with_1_constant_input(self):
        """
        TODO: Same as above test case: Rather make input2 have placeholder_with_default so that we don't need to
        TODO: provide a value for it necessarily.

        Adds a single component with 2-to-1 graph_fn to the core, and the second input to the
        graph_fn is already blocked by the component.
        """
        component = Dummy2To1(scope="dummy")
        test = ComponentTest(component=component, input_spaces=dict(input1=int, input2=int))

        # Expected output: in1 + (const 1.0)
        test.test(("run", [4, 5]), expected_outputs=9)
