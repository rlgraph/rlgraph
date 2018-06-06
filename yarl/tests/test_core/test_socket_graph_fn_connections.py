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

from yarl.components import Component, CONNECT_INS, CONNECT_OUTS
from yarl.tests import ComponentTest, root_logger
from tests.dummy_components import Dummy1to1, Dummy2to1, Dummy1to2, Dummy0to1


class TestSocketGraphFnConnections(unittest.TestCase):
    """
    Tests for different ways to connect Sockets to other Sockets (of other Components) and GraphFunctions.
    """
    root_logger.setLevel(level=logging.INFO)

    def test_1to1_component(self):
        """
        Adds a single component with 1-to-1 graph_fn to the core and passes a value through it.
        """
        component = Dummy1to1(scope="dummy")
        test = ComponentTest(component=component, input_spaces=dict(input=float))

        # expected output: input + 1.0
        test.test(out_socket_names="output", inputs=1.0, expected_outputs=2.0)
        test.test(out_socket_names="output", inputs=-5.0, expected_outputs=-4.0)

    def test_2to1_component(self):
        """
        Adds a single component with 2-to-1 graph_fn to the core and passes 2 values through it.
        """
        component = Dummy2to1(scope="dummy")
        test = ComponentTest(component=component, input_spaces=dict(input1=float, input2=float))

        # expected output: input1 + input2
        test.test(out_socket_names="output", inputs=dict(input1=1.0, input2=2.9), expected_outputs=3.9)
        test.test(out_socket_names="output", inputs=dict(input1=4.9, input2=-0.1),
                  expected_outputs=np.array(4.8, dtype=np.float32))

    def test_1to2_component(self):
        """
        Adds a single component with 1-to-2 graph_fn to the core and passes a value through it.
        """
        component = Dummy1to2(scope="dummy")
        test = ComponentTest(component=component, input_spaces=dict(input=float))

        # expected outputs: (input, input+1.0)
        test.test(out_socket_names=["output1", "output2"], inputs=1.0, expected_outputs=[1.0, 2.0])
        test.test(out_socket_names=["output1", "output2"], inputs=4.5, expected_outputs=[4.5, 5.5])

    def test_0to1_component(self):
        """
        Adds a single component with 0-to-1 graph_fn to the core and passes a value through it.
        """
        component = Dummy0to1(scope="dummy")
        test = ComponentTest(component=component, input_spaces=dict(input=float))

        # expected outputs: Always 0.0
        test.test(out_socket_names=["output"], inputs=None, expected_outputs=[8.0])

    def test_2to1_component_with_constant_input_value(self):
        """
        Adds a single component with 1-to-1 graph_fn to the core and blocks the input with a constant value.
        """
        component = Dummy2to1(scope="dummy")
        test = ComponentTest(component=component, input_spaces=dict(input1=5.5, input2=float))

        # expected output: (const 5.5) + in2 = 10.0
        test.test(out_socket_names="output", inputs=dict(input2=4.5), expected_outputs=10.0)

    def test_1to1_to_2to1_component_with_constant_input_value(self):
        """
        Adds two components in sequence, 1-to-1 and 2-to-1, to the core and blocks one of the inputs of 2-to-1
        with a constant value (so that this constant value is not at the border of the core component).
        """
        core = Component(scope="container")
        sub_comp1 = Dummy1to1(scope="A")
        sub_comp2 = Dummy2to1(scope="B")
        core.add_component(sub_comp1, connections=CONNECT_INS)
        core.add_component(sub_comp2, connections=CONNECT_OUTS)
        core.connect(1.1, (sub_comp2, "input1"))
        core.connect((sub_comp1, "output"), (sub_comp2, "input2"))

        test = ComponentTest(component=core, input_spaces=dict(input=float))

        # expected output: (input + 1.0) + 1.1
        test.test(out_socket_names="output", inputs=78.4, expected_outputs=80.5)
        test.test(out_socket_names="output", inputs=-5.2, expected_outputs=-3.1)

    def test_connecting_two_1to1_components(self):
        """
        Adds two components with 1-to-1 graph_fns to the core, connects them and passes a value through it.
        """
        core = Component(scope="container")
        sub_comp1 = Dummy1to1(scope="comp1")
        sub_comp2 = Dummy1to1(scope="comp2")
        core.add_component(sub_comp1, connections=CONNECT_INS)
        core.add_component(sub_comp2, connections=CONNECT_OUTS)
        core.connect(sub_comp1, sub_comp2)

        test = ComponentTest(component=core, input_spaces=dict(input=float))

        # expected output: input + 1.0 + 1.0
        test.test(out_socket_names="output", inputs=1.1, expected_outputs=3.1)
        test.test(out_socket_names="output", inputs=-5.1, expected_outputs=-3.1)

    def test_connecting_1to2_to_2to1(self):
        """
        Adds two components with 1-to-2 and 2-to-1 graph_fns to the core, connects them and passes a value through it.
        """
        core = Component(scope="container")
        sub_comp1 = Dummy1to2(scope="comp1")  # outs=in,in+1
        sub_comp2 = Dummy2to1(scope="comp2")  # out =in1+in2
        core.add_component(sub_comp1, connections=CONNECT_INS)
        core.add_component(sub_comp2, connections=CONNECT_OUTS)
        core.connect(sub_comp1, sub_comp2)

        test = ComponentTest(component=core, input_spaces=dict(input=float))

        # expected output: input + (input + 1.0)
        test.test(out_socket_names="output", inputs=100.9, expected_outputs=np.float32(202.8))
        test.test(out_socket_names="output", inputs=-5.1, expected_outputs=np.float32(-9.2))

    def test_connecting_in1_and_1to1_to_1to1_no_labels(self):
        """
        Adds two components (A, B) with 1-to-1 graph_fns to the core.
        Connects "input1" with A, A's "output" to "output".
        Connects "input2" with B and B's output to A.
        If we now pull out-Socket "output", it should know by the given input, which op we actually want.
        """
        core = Component(inputs=["input1", "input2"], outputs="output", scope="container")
        a = Dummy1to1(scope="A")
        b = Dummy1to1(scope="B")
        # Throw in the sub-components.
        core.add_components(a, b)
        # Connect them.
        core.connect("input1", (a, "input"))
        core.connect("input2", (b, "input"))
        core.connect((b, "output"), (a, "input"))  # a/input now has two incoming connections.
        core.connect((a, "output"), "output")

        test = ComponentTest(component=core, input_spaces=dict(input1=float, input2=float))

        # Now pulling on the same Socket (given one of the input Sockets) should trigger the correct
        # op.
        # expected output: input1 + 1.0(a)
        test.test(out_socket_names="output", inputs=dict(input1=np.array(1.1)), expected_outputs=2.1)
        # expected output: input2 + 1.0(b) + 1.0(a)
        test.test(out_socket_names="output", inputs=dict(input2=np.array(1.1, dtype=np.float32)), expected_outputs=3.1)

    def test_connecting_in1_to_1to1_no_labels(self):
        """
        Adds one component (A) with 1-to-1 graph_fn to the core which has two in-Sockets and 1 out-Socket.
        Connects "input_c" with A, A's "output" to "output".
        Connects "input_a" with A, A's "output" to "output".
        No labels whatsoever.
        If we now pull "output", and provide input1 AND input2, it should use the in-Socket that comes
        first in alphabetic order ("input_a" even if it's defined second).
        """
        core = Component(inputs=["input_c", "input_a"], outputs="output", scope="container")
        a = Dummy1to1(scope="A")
        # Throw in the sub-component.
        core.add_components(a)
        # Connect correctly.
        core.connect("input_c", (a, "input"))
        core.connect("input_a", (a, "input"))
        core.connect((a, "output"), "output")

        test = ComponentTest(component=core, input_spaces=dict(input_c=float, input_a=float))

        # Now pulling on "output" and providing both inputs should cause disambiguity, but it should chose
        # the one fist in alphabetic order (input_a).
        # expected output: input_a + 1.0
        test.test(out_socket_names="output", inputs=dict(input_c=np.array(1.5), input_a=np.array(2.1)),
                  expected_outputs=3.1)

    def test_connecting_in1_to_1to1_with_labels(self):
        """
        TODO: labels not being passed on from Socket to next Socket!
        Same as `test_connecting_in1_to_1to1_no_labels` but with labels.
        So if we provide both inputs, it should know which one to take (instead of using alphabetic order).
        """
        core = Component(inputs=["input_c", "input_a"], outputs="output", scope="container")
        a = Dummy1to1(scope="A")
        # Throw in the sub-component.
        core.add_components(a)
        # Connect correctly and with labels.
        core.connect("input_c", (a, "input"), label="from_in_c")
        core.connect("input_a", (a, "input"), label="from_in_a")
        # force using the input_c path (over the alphabetically favored input_a).
        core.connect((a, "output"), "output", label="from_in_c")

        test = ComponentTest(component=core, input_spaces=dict(input_c=float, input_a=float))

        # Now pulling on "output" and providing both inputs should cause disambiguity, but it should chose
        # the one fist in alphabetic order (input_a).
        # expected output: input_a + 1.0
        test.test(out_socket_names="output", inputs=dict(input_c=np.array(1.5), input_a=np.array(2.1)),
                  expected_outputs=2.5)

