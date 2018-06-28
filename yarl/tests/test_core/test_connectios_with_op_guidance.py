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

from yarl.components import Component
from yarl.tests import ComponentTest
from yarl.utils import root_logger
from yarl.tests.dummy_components import Dummy1to1, DummyWithSubComponents


class TestConnectionsWithOpGuidance(unittest.TestCase):
    """
    Tests for connecting sub-components via detailed op-guidance.
    """
    root_logger.setLevel(level=logging.INFO)

    def test_single_component(self):
        """
        'A' is 1to1: send "input" through A, receive output.
        """
        a = Dummy1to1(scope="A")
        test = ComponentTest(component=a, input_spaces=dict(run=float))
        test.test(api_method="run", params=np.array(1.1), expected_outputs=0.0)

    def test_component_with_sub_component(self):
        a = DummyWithSubComponents(scope="A")
        test = ComponentTest(component=a, input_spaces=dict(run=float))

        test.test(api_method="run", params=np.array(1.1), expected_outputs=0.0)

    def test_simple_diamond_sub_component_setup(self):
        """
        Adds 4 sub-components (A, B, C, D) with 1-to-1 graph_fns to the core.
        in1 -> A (like preprocessor in DQN)
        in2 -> A
        A -> B (like policy in DQN)
        A -> C (like target policy in DQN)
        B -> Din1 (like loss func in DQN: q_vals_s)
        C -> Din2 (q_vals_sp)
        """
        container = Component(inputs=["input1", "input2"], outputs=["output"], scope="container")
        a = Dummy1to1(scope="A")
        b = Dummy1to1(scope="B")
        c = Dummy1to1(scope="C")
        d = Dummy2to1(scope="D")

        # Throw in the sub-components.
        container.add_components(a, b, c, d)

        # Connect them on detailed op-level (see above for connection details).
        in1_through_a = a("input1")  # send input1 through Component A
        in2_through_a = a("input2")  # same with input2
        # "Manually" split the 2 ops coming out of A: 2x->B and 1x->C.
        b_out_1 = b(in1_through_a)
        b_out_2 = b(in2_through_a)
        c_out = c(in2_through_a)
        # Merge b_out and c_out again into D (in1 and in2 Sockets).
        final_1 = d(b_out_1, c_out)
        final_2 = d(b_out_2, c_out)

        container.connect(d["output"], "output")

        test = ComponentTest(component=container, input_spaces=dict(input1=float, input2=float))

        # Push both api_methods through graph to receive correct (single-op) output calculation.
        test.test(out_socket_names="output", inputs=dict(input1=np.array(1.1), input2=np.array(0.5)),
                  expected_outputs=0.0)

