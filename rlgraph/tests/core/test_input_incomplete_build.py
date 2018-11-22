# Copyright 2018 The Rlgraph Authors, All Rights Reserved.
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
from rlgraph.utils import root_logger, RLGraphError, RLGraphBuildError
from rlgraph.tests.dummy_components_with_sub_components import *


class TestInputIncompleteTest(unittest.TestCase):
    """
    Tests for different scenarios, where a model is build, but pieces remain input-incomplete (these should be
    reported then by meaningful error messages).
    """
    root_logger.setLevel(level=logging.INFO)

    def test_single_component_with_single_api_method(self):
        """
        Component cannot be built due to its sub-component remaining input incomplete.
        """
        a = DummyProducingInputIncompleteBuild(scope="A")
        try:
            test = ComponentTest(component=a, input_spaces=dict(input_=float))
        except RLGraphBuildError as e:
            print("Seeing expected RLGraphBuildError. Test ok.")
        else:
            raise RLGraphError("Not seeing expected RLGraphBuildError with input-incomplete model!")
