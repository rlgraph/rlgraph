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

from yarl.components import Component, CONNECT_INS, CONNECT_OUTS
from yarl.tests import ComponentTest, root_logger
from tests.dummy_components import Dummy1to1, Dummy2to1, Dummy1to2, Dummy0to1


class TestDQNAgent(unittest.TestCase):
    """
    Tests the DQN Agent on simple deterministic learning tasks.
    """
    root_logger.setLevel(level=logging.INFO)

    def test_dqn_on_2x2_grid_world(self):
        """
        Creates a DQNAgent and runs it via a Runner on a simple 2x2 GridWorld.
        """
        component = Dummy1to1(scope="dummy")

        # Expected output: input + 1.0
        test.test(out_socket_names="output", inputs=1.0, expected_outputs=2.0)
        test.test(out_socket_names="output", inputs=-5.0, expected_outputs=-4.0)
