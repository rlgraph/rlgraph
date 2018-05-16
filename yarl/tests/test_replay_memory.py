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
    capacity = 10

    def test_insert_retrieve(self):
        """
        Test simple insert and retrieval of data.
        """
        memory = ReplayMemory(
            capacity=self.capacity,
            next_states=True
        )
        test = ComponentTest(component=memory, input_spaces=dict(records=self.record_space))

        # Run the test.
        observation = self.record_space.sample(size=(1,))
        print(observation)
        result = test.test(out_socket_name="insert", inputs=observation, expected_outputs=[])

    def test_insert_after_full(self):
        memory = ReplayMemory(
            capacity=self.capacity,
            next_states=True
        )
        test = ComponentTest(component=memory, input_spaces=dict(records=self.record_space))
        buffer_size, buffer_index = memory.get_variables()

        size_value, index_value = test.get_variable_values([buffer_size, buffer_index])
        # Assert indices 0 before insert.
        self.assertTrue(size_value == 0)
        self.assertTrue(index_value == 0)

        # insert one more element than capacity
        observation = self.record_space.sample(size=self.capacity + 1)
        result = test.test(out_socket_name="insert", inputs=observation, expected_outputs=[])

        # TODO fetch index variables here
        size_value, index_value = test.get_variable_values([buffer_size, buffer_index])
        # Size should be full now
        self.assertTrue(size_value == self.capacity)
        # One over capacity
        self.assertTrue(index_value == 1)