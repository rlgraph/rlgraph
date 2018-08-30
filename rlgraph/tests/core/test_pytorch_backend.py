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


class TestPytorchBackend(unittest.TestCase):
    """
    Tests PyTorch component execution.

    # TODO: This is a temporary test. We will later run all backend-specific
    tests via setting the executor in the component-test.
    """
    root_logger.setLevel(level=logging.INFO)

    def test_api_call_no_variables(self):
        """
        Tests define-by-run call of api method via defined_api method on a
        component without variables.
        """
        a = Dummy2To1()
        test = ComponentTest(component=a, input_spaces=dict(input1=float, input2=float), backend="pytorch")
        test.test(("run", [1.0, 2.0]), expected_outputs=3.0, decimals=4)

    def test_2to1_component(self):
        """
        Adds a single component with 2-to-1 graph_fn to the core and passes 2 values through it.
        """
        component = Dummy2To1(scope="dummy")
        test = ComponentTest(component=component, input_spaces=dict(input1=float, input2=float))

        # Expected output: input1 + input2
        result = test.test(("run", [1.0, 2.9]), expected_outputs=3.9)
        #test.test(("run", [4.9, -0.1]), expected_outputs=np.array(4.8, dtype=np.float32))