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

import unittest
import numpy as np

from yarl.components.layers import Stack
from yarl.tests.dummy_components import Dummy1To1, Dummy2To1, Dummy1To2
from yarl.spaces import FloatBox, IntBox
from yarl.tests import ComponentTest


class TestStack(unittest.TestCase):
    """
    Tests for Stack Components of different length (number of sub-Components).
    """
    def test_one_sub_component(self):
        stack = Stack(Dummy1To1(constant_value=3.0), api_methods={"run"})
        test = ComponentTest(component=stack, input_spaces=dict(run=FloatBox()))

        test.test(api_method="run", params=4.6, expected_outputs=7.6)

