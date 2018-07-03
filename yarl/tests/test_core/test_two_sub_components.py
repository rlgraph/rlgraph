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
from yarl.tests.dummy_components import *


class TestTwoSubComponents(unittest.TestCase):
    """
    Tests for different ways to place two sub-Components into the core.
    """
    root_logger.setLevel(level=logging.INFO)

    # TODO do we want this test class or are they covered by 'test_connections_with_op_guidance.py'?

    def test_connecting_two_1to1_components(self):
        """
        Adds two components with 1-to-1 graph_fns to the core, connects them and passes a value through it.
        """
        core = Component(scope="container")
        sub_comp1 = Dummy1To1(scope="comp1")
        sub_comp2 = Dummy1To1(scope="comp2")
        core.add_components(sub_comp1)
        core.add_components(sub_comp2)

        test = ComponentTest(component=core)

        # Expected output: input + 1.0 + 1.0
        test.test(api_method="run", params=1.1, expected_outputs=3.1)
        test.test(api_method="run", params=-5.1, expected_outputs=-3.1)

    def test_connecting_1to2_to_2to1(self):
        """
        Adds two components with 1-to-2 and 2-to-1 graph_fns to the core, connects them and passes a value through it.
        """
        core = Component(scope="container")
        sub_comp1 = Dummy2To1(scope="comp1")  # outs=in,in+1
        sub_comp2 = Dummy2To1(scope="comp2")  # out =in1+in2
        core.add_components(sub_comp1)
        core.add_components(sub_comp2)

        test = ComponentTest(component=core, input_spaces=dict(input=float))

        # Expected output: input + (input + 1.0)
        test.test(api_method="run", params=100.9, expected_outputs=np.float32(202.8))
        test.test(api_method="run", params=-5.1, expected_outputs=np.float32(-9.2))

    def test_1to1_to_2to1_component_with_constant_input_value(self):
        """
        Adds two components in sequence, 1-to-1 and 2-to-1, to the core and blocks one of the api_methods of 2-to-1
        with a constant value (so that this constant value is not at the border of the core component).
        """
        core = Component(scope="container")
        sub_comp1 = Dummy1To1(scope="A")
        sub_comp2 = Dummy2To1(scope="B")
        core.add_components(sub_comp1)
        core.add_components(sub_comp2)

        test = ComponentTest(component=core, input_spaces=dict(input=float))

        # Expected output: (input + 1.0) + 1.1
        test.test(api_method="run", params=78.4, expected_outputs=80.5)
        test.test(api_method="run", params=-5.2, expected_outputs=-3.1)
