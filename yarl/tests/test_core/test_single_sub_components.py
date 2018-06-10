# Copyright 2018 The YARL-Project, All Rights Reserved.
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

from yarl.tests import ComponentTest
from yarl.utils import root_logger
from yarl.tests.dummy_components import Dummy1to1, Dummy2to1, Dummy1to2, Dummy0to1, Dummy2to1Where1ConnectedWithConstant


class TestSingleSubComponents(unittest.TestCase):
    """
    Tests for different ways to place different, but single sub-Components into the core.
    """
    root_logger.setLevel(level=logging.INFO)

    def test_1to1_component(self):
        """
        Adds a single component with 1-to-1 graph_fn to the core and passes a value through it.
        """
        component = Dummy1to1(scope="dummy")
        test = ComponentTest(component=component, input_spaces=dict(input=float))

        # Expected output: input + 1.0
        test.test(out_socket_names="output", inputs=1.0, expected_outputs=2.0)
        test.test(out_socket_names="output", inputs=-5.0, expected_outputs=-4.0)

    def test_2to1_component(self):
        """
        Adds a single component with 2-to-1 graph_fn to the core and passes 2 values through it.
        """
        component = Dummy2to1(scope="dummy")
        test = ComponentTest(component=component, input_spaces=dict(input1=float, input2=float))

        # Expected output: input1 + input2
        test.test(out_socket_names="output", inputs=dict(input1=1.0, input2=2.9), expected_outputs=3.9)
        test.test(out_socket_names="output", inputs=dict(input1=4.9, input2=-0.1),
                  expected_outputs=np.array(4.8, dtype=np.float32))

    def test_1to2_component(self):
        """
        Adds a single component with 1-to-2 graph_fn to the core and passes a value through it.
        """
        component = Dummy1to2(scope="dummy")
        test = ComponentTest(component=component, input_spaces=dict(input=float))

        # Expected outputs: (input, input+1.0)
        test.test(out_socket_names=["output1", "output2"], inputs=1.0, expected_outputs=[1.0, 2.0])
        test.test(out_socket_names=["output1", "output2"], inputs=4.5, expected_outputs=[4.5, 5.5])

    def test_0to1_component(self):
        """
        Adds a single component with 0-to-1 graph_fn to the core and passes a value through it.
        """
        component = Dummy0to1(scope="dummy")
        test = ComponentTest(component=component, input_spaces=dict(input=float))

        # Expected outputs: Always 0.0
        test.test(out_socket_names=["output"], inputs=None, expected_outputs=[8.0])

    def test_2to1_component_with_constant_input_value(self):
        """
        Adds a single component with 1-to-1 graph_fn to the core and blocks the input with a constant value.
        """
        component = Dummy2to1(scope="dummy")
        test = ComponentTest(component=component, input_spaces=dict(input1=5.5, input2=float))

        # Expected output: (const 5.5) + in2
        test.test(out_socket_names="output", inputs=dict(input2=4.5), expected_outputs=10.0)

    def test_2to1_component_with_constant_graph_fn_input1(self):
        """
        Adds a single component with 2-to-1 graph_fn to the core, and the second input to the
        graph_fn is already blocked by the component.
        """
        component = Dummy2to1Where1ConnectedWithConstant(scope="dummy")
        test = ComponentTest(component=component, input_spaces=dict(input1=float))

        # Expected output: in1 + (const 3.0)
        test.test(out_socket_names="output", inputs=dict(input1=4.5), expected_outputs=7.5)
