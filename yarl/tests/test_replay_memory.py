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

from yarl.components.memories.replay_memory import ReplayMemory
from yarl.spaces import Dict
from yarl.tests import ComponentTest

import numpy as np


class TestReplayMemory(unittest.TestCase):
    """
    Tests sampling and insertion behaviour of the replay_memory module.
    """

    record_space = Dict(
        state=Dict(state1=float, state2=float),
        action=Dict(action1=float),
        reward=float,
        terminal=int
    )

    def test_insert_retrieve(self):
        component_to_test = ReplayMemory(
            record_space=self.record_space,
            capacity=10,
            next_states=True
        )

        test = ComponentTest(component=component_to_test, input_spaces=dict(input=self.record_space))

        # Run the test.
        input_ = np.array([[0.5, 2.0]])
        expected = np.array([[2.5, 2.5]])

        result = test.test(out_socket_name="output", inputs=input_, expected_outputs=expected)
        self.assertTrue(result)

