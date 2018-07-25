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

from rlgraph.components import Component
from rlgraph.tests import ComponentTest
from rlgraph.utils import root_logger
from rlgraph.tests.dummy_components import Dummy1To1


class TestConnectionsWithLabels(unittest.TestCase):
    """
    Tests for different ways to place different, but single sub-Components into the core.
    """
    root_logger.setLevel(level=logging.INFO)

    # TODO: Remove this test class?

    # def test_connecting_in1_and_1to1_to_1to1_no_labels(self):
    #     """
    #     Adds two components (A, B) with 1-to-1 graph_fns to the core.
    #     Connects "input1" with A, A's "output" to "output".
    #     Connects "input2" with B and B's output to A.
    #     If we now pull out-Socket "output", it should know by the given input, which op we actually want.
    #     """
    #     core = Component(inputs=["input1", "input2"], outputs="output", scope="container")
    #     a = Dummy1to1(scope="A")
    #     b = Dummy1to1(scope="B")
    #     # Throw in the sub-components.
    #     core.add_components(a, b)
    #     # Connect them.
    #     core.connect("input1", (a, "input"))
    #     core.connect("input2", (b, "input"))
    #     core.connect((b, "output"), (a, "input"))  # a/input now has two incoming connections.
    #     core.connect((a, "output"), "output")
    #
    #     test = ComponentTest(component=core, input_spaces=dict(input1=float, input2=float))
    #
    #     # Now pulling on the same Socket (given one of the input Sockets) should trigger the correct op.
    #     # Expected output: input1 + 1.0(a)
    #     test.test(out_socket_names="output", inputs=dict(input1=np.array(1.1)), expected_outputs=2.1)
    #     # Expected output: input2 + 1.0(b) + 1.0(a)
    #     test.test(out_socket_names="output", inputs=dict(input2=np.float32(1.1)), expected_outputs=3.1)
    #
    # def test_connecting_in1_to_1to1_no_labels(self):
    #     """
    #     Adds one component (A) with 1-to-1 graph_fn to the core which has two in-Sockets and 1 out-Socket.
    #     Connects "input_c" with A, A's "output" to "output".
    #     Connects "input_a" with A, A's "output" to "output".
    #     No labels whatsoever.
    #     If we now pull "output", and provide input1 AND input2, it should use the in-Socket that comes
    #     first in alphabetic order ("input_a" even if it's defined second).
    #     """
    #     core = Component(inputs=["input_c", "input_a"], outputs="output", scope="container")
    #     a = Dummy1to1(scope="A")
    #     # Throw in the sub-component.
    #     core.add_components(a)
    #     # Connect correctly.
    #     core.connect("input_c", (a, "input"))
    #     core.connect("input_a", (a, "input"))
    #     core.connect((a, "output"), "output")
    #
    #     test = ComponentTest(component=core, input_spaces=dict(input_c=float, input_a=float))
    #
    #     # Now pulling on "output" and providing both api_methods should cause disambiguity, but it should chose
    #     # the one fist in alphabetic order (input_a).
    #     # Expected output: input_a + 1.0
    #     test.test(out_socket_names="output", inputs=dict(input_c=np.array(1.5), input_a=np.array(2.1)),
    #               expected_outputs=3.1)
    #
    # def test_connecting_in1_and_in2_to_1to1_to_out1_and_out2_with_labels(self):
    #     """
    #     Same as `test_connecting_in1_to_1to1_no_labels` but with labels.
    #     So if we provide both api_methods, it should know which one to take (instead of using alphabetic order).
    #     """
    #     core = Component(inputs=["input_c", "input_a"], outputs="output", scope="container")
    #     dummy = Dummy1to1(scope="dummy")
    #     # Throw in the sub-component.
    #     core.add_components(dummy)
    #     # Connect correctly (with labels).
    #     # [input_c->dummy/input] is now labelled as "from_in_c" (op_records passing to dummy will be labelled so).
    #     core.connect("input_c", (dummy, "input"), label="from_in_c")
    #     # [input_a->dummy/input] is now labelled as "from_in_a" (op_records passing to dummy will be labelled so).
    #     core.connect("input_a", (dummy, "input"), label="from_in_a")
    #     # Force using the input_c path (over the alphabetically favored input_a).
    #     # [dummy/output->output] is now labelled as "from_in_c" and will thus only allow ops that have this label to
    #     # be passed through to "output" (all other op_records will be filtered).
    #     core.connect((dummy, "output"), "output", label="from_in_c")
    #
    #     test = ComponentTest(component=core, input_spaces=dict(input_c=float, input_a=float))
    #
    #     # Now pulling on "output" and providing both api_methods should not cause disambiguity since out out-Socket
    #     # was connected to A's "output" through the label "from_in_c", so it should always use "input_c".
    #     # Expected output: input_c + 1.0
    #     test.test(out_socket_names="output", inputs=dict(input_c=np.array(1.5), input_a=np.array(2.1)),
    #               expected_outputs=2.5)
    #     test.test(out_socket_names="output", inputs=dict(input_c=np.array(1.5), input_a=np.array(2.1)),
    #               expected_outputs=2.5)
    #
    # def test_connecting_in_to_2x_to_different_1to1_then_2x_to_1to1_to_out1_and_out2_with_labels(self):
    #     """
    #
    #     """
    #     core = Component(inputs="input", outputs=["output1", "output2"], scope="container")
    #     pre1 = Dummy1to1(scope="pre1")
    #     pre2 = Dummy1to1(scope="pre2", constant_value=2.0)  # make it different from pre1
    #     hub = Dummy1to1(scope="hub")
    #
    #     # Throw in the sub-components.
    #     core.add_components(pre1, pre2, hub)
    #
    #     # Connect correctly (with labels).
    #     core.connect("input", (pre1, "input"))
    #     core.connect("input", (pre2, "input"))
    #     # [pre1/input->hub/input] is now labelled as "from_pre1" (op_records passing to dummy will be labelled so).
    #     core.connect((pre1, "output"), (hub, "input"), label="from_pre1")
    #     # [pre2/input->hub/input] is now labelled as "from_pre2" (op_records passing to dummy will be labelled so).
    #     core.connect((pre2, "output"), (hub, "input"), label="from_pre2")
    #     # Provide both paths (via pre1 or pre2) via labels for the two different out-Sockets.
    #     core.connect((hub, "output"), "output1", label="from_pre1")
    #     core.connect((hub, "output"), "output2", label="from_pre2")
    #
    #     test = ComponentTest(component=core, input_spaces=dict(input=float))
    #
    #     # Now pulling on "output1", should take the op over pre1. Pulling "output2" should take the op over pre2.
    #     # Expected output: (input + 1.0) + 1.0
    #     test.test(api_method="output1", inputs=dict(input=np.array(187.5)), expected_outputs=189.5)
    #     # Expected output: (input + 2.0) + 1.0
    #     test.test(api_method="output2", inputs=dict(input=np.array(87.5)), expected_outputs=90.5)
