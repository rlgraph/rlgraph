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
import unittest

from rlgraph.tests import ComponentTest
from rlgraph.utils import root_logger
from rlgraph.tests.dummy_components import *


class TestDevicePlacements(unittest.TestCase):
    """
    Tests different ways to place Components and their ops/variables on different devices.
    """
    root_logger.setLevel(level=logging.INFO)

    def test_single_component(self):
        """
        Place the entire Component on its own device.
        """
        a = Dummy1To1(scope="A", device="/device:CPU:0")
        test = ComponentTest(component=a, input_spaces=dict(input_=float))
        # Actually check the device of the ops in a.
        self.assertEqual(a.api_methods["run"].in_op_columns[0].op_records[0].op.device, "/device:CPU:0")
        self.assertEqual(a.api_methods["run"].out_op_columns[0].op_records[0].op.device, "/device:CPU:0")
        # Expected: in + 1.0
        test.test(("run", 1.1), expected_outputs=2.1)

    def test_single_component_with_variables(self):
        """
        Place variables on CPU, ops on GPU (if exists).
        """
        var_device = "/device:CPU:0"
        op_device = "/device:GPU:0"
        a = DummyWithVar(scope="A", device=dict(variables=var_device, ops=op_device))
        test = ComponentTest(component=a, input_spaces=dict(input_=float))

        # Vars -> CPU.
        self.assertEqual(a.variables["A/constant-variable"].device, var_device)
        # Placeholders -> GPU.
        self.assertEqual(a.api_methods["run_plus"].in_op_columns[0].op_records[0].op.device, op_device)
        self.assertEqual(a.api_methods["run_minus"].in_op_columns[0].op_records[0].op.device, op_device)
        # Actual ops -> GPU.
        self.assertEqual(a.api_methods["run_plus"].out_op_columns[0].op_records[0].op.device, op_device)
        self.assertEqual(a.api_methods["run_minus"].out_op_columns[0].op_records[0].op.device, op_device)

        # Expected: in + 2.0
        test.test(("run_plus", 1.1), expected_outputs=3.1)

    def test_sub_components_with_device_map(self):
        """
        Place variables on CPU, ops on GPU (if exists).
        """
        a = DummyWithSubComponents(scope="A")
        comp_device = "/device:GPU:0"
        sub_comp_device = "/device:CPU:0"
        test = ComponentTest(component=a, input_spaces=dict(input_=float),
                             device_strategy="custom",
                             device_map=dict({"A/dummy-with-var": sub_comp_device, "A": comp_device}))
        # Actually check the device of the variables and ops in a.
        actual_comp_device = "/device:GPU:0" if "/device:GPU:0" in test.graph_builder.available_devices else \
            "/device:CPU:0"
        self.assertEqual(a.api_methods["run1"].in_op_columns[0].op_records[0].op.device, actual_comp_device)
        self.assertEqual(a.api_methods["run1"].out_op_columns[0].op_records[0].op.device, sub_comp_device)
        self.assertEqual(a.api_methods["run1"].out_op_columns[0].op_records[1].op.device, actual_comp_device)
        self.assertEqual(a.api_methods["run2"].in_op_columns[0].op_records[0].op.device, actual_comp_device)
        self.assertEqual(a.api_methods["run2"].out_op_columns[0].op_records[0].op.device, actual_comp_device)

        test.test(("run1", 1.1), expected_outputs=[3.1, 4.1])
        test.test(("run2", -1.1), expected_outputs=-2.1)

