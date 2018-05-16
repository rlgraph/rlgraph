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
from yarl.spaces import Dict, Discrete, IntBox
from yarl.tests import ComponentTest


class TestReplayMemory(unittest.TestCase):
    """
    Tests sampling and insertion behaviour of the replay_memory module.
    """

    record_space = Dict(
        states=dict(state1=float, state2=float),
        actions=dict(action1=float),
        reward=float,
        terminal=IntBox(shape=(), low=0, high=1),
        add_batch_rank=True
    )

    def test_insert_retrieve(self):
        """
        Test simple insert and retrieval of data.
        """
        component_to_test = ReplayMemory(
            capacity=10,
            next_states=True
        )
        test = ComponentTest(component=component_to_test, input_spaces=dict(records=self.record_space))

        # Run the test.
        observation = self.record_space.sample()
        print(observation)
        result = test.test(out_socket_name="insert", inputs=observation, expected_outputs=[])

    # def test_insert_after_full(self):
    #     component_to_test = ReplayMemory(
    #         capacity=10,
    #         next_states=True
    #     )
    #     test = ComponentTest(component=component_to_test, input_spaces=dict(records=self.record_space))
    #
    #     # Run the test.
    #     observation = self.record_space.sample(size=1)
    #     result = test.test(out_socket_name="insert", inputs=observation, expected_outputs=[])