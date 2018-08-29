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

    def test_component_with_sub_component(self):
        a = DummyWithSubComponents(scope="A")
        test = ComponentTest(component=a, input_spaces=dict(input_=float),
                             backend="pytorch")

        # Expected: (1): in + 2.0  (2): [result of (1)] + 1.0
        test.test(("run1", 1.1), expected_outputs=[3.1, 4.1], decimals=4)
        # Expected: in - 2.0 + 1.0
        test.test(("run2", 1.1), expected_outputs=0.1, decimals=4)
