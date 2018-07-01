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
from yarl.tests.dummy_components import Dummy1To1, Dummy2To1, DummyWithSubComponents


class TestConnectionsWithOpGuidance(unittest.TestCase):
    """
    Tests for connecting sub-components via detailed op-guidance.
    """
    root_logger.setLevel(level=logging.INFO)

    def test_single_component_with_single_api_method(self):
        """
        'A' is 1to1: send "input" through A, receive output.
        """
        a = Dummy1To1(scope="A")
        test = ComponentTest(component=a, input_spaces=dict(run=float))
        # Expected: in + 1.0
        test.test(api_method="run", params=np.array(1.1), expected_outputs=2.1)

    def test_component_with_sub_component(self):
        a = DummyWithSubComponents(scope="A")
        test = ComponentTest(component=a, input_spaces=dict(run1=float, run2=float))

        # Expected: (1): in + 2.0  (2): [result of (1)] + 1.0
        test.test(api_method="run1", params=np.array(1.1), expected_outputs=(3.1, 4.1))

    def test_diamond_4x_sub_component_setup(self):
        """
        Adds 4 sub-components (A, B, C, D) with 1-to-1 graph_fns to the core.
        in1 -> A (like preprocessor in DQN)
        in2 -> A
        A -> B (like policy in DQN)
        A -> C (like target policy in DQN)
        B -> Din1 (like loss func in DQN: q_vals_s)
        C -> Din2 (q_vals_sp)
        """
        container = Component(scope="container")
        a = Dummy1To1(scope="A")
        b = Dummy1To1(scope="B")
        c = Dummy1To1(scope="C")
        d = Dummy2To1(scope="D")

        # Throw in the sub-components.
        container.add_components(a, b, c, d)

        # Define container's API:
        def run(self_, input1, input2):
            """
            Describes the diamond setup in1->A->B; in2->A->C; C,B->D->output
            """
            in1_past_a = self_.call(self_.sub_components["A"].run, input1)
            in2_past_a = self_.call(self_.sub_components["A"].run, input2)
            past_b = self_.call(self_.sub_components["B"].run, in1_past_a)
            past_c = self_.call(self_.sub_components["C"].run, in2_past_a)
            past_d = self_.call(self_.sub_components["D"].run, past_b, past_c)
            return past_d

        container.define_api_method("run", run)

        test = ComponentTest(component=container, input_spaces=dict(run=(float, float)))

        # Push both api_methods through graph to receive correct (single-op) output calculation.
        test.test(api_method="run", params=(np.array(1.1), np.array(0.5)), expected_outputs=(0.0, 0.0))

