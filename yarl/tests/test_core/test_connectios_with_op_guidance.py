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
from yarl.tests.dummy_components import Dummy1to1, Dummy2to1


class TestConnectionsWithOpGuidance(unittest.TestCase):
    """
    Tests for connecting sub-components via detailed op-guidance.
    """
    root_logger.setLevel(level=logging.INFO)

    def test_connecting_in1_and_1to1_to_1to1_no_labels(self):
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
        in1_through_a = a("input1")  # send input1 through Component "a"
        in2_through_a = a("input2")  # same with input2
        # "Manually" split the 2 ops coming out of a via b and c.
        b_out = b(in1_through_a)
        c_out = c(in2_through_a)
        # Merge b_out and c_out again into D (in1 and in2 Sockets).
        final = d(b_out, c_out)
        container.define_outputs("output", final)  # TODO:

        test = ComponentTest(component=container, input_spaces=dict(input1=float, input2=float))

        # Push both inputs through graph to receive correct (single-op) output calculation.
        test.test(out_socket_names="output", inputs=dict(input1=np.array(1.1), input2=np.array(0.5)),
                  expected_outputs=0.0)

