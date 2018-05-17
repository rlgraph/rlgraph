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
from yarl.spaces import Dict, IntBox
from yarl.tests import ComponentTest


class TestRingBufferMemory(unittest.TestCase):
    """
    Tests the ring buffer. The ring buffer has very similar tests to
    the replay memory as it supports similar insertion and retrieval semantics,
    but needs additional tests on episode indexing and its latest semantics.
    """

    record_space = Dict(
        states=dict(state1=float, state2=float),
        actions=dict(action1=float),
        reward=float,
        terminal=IntBox(low=0, high=1),
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
        test = ComponentTest(component=memory, input_spaces=dict(
            records=self.record_space,
            num_records=int
        ))

        # Run the test.
        observation = self.record_space.sample(size=5)
        test.test(out_socket_name="insert", inputs=observation, expected_outputs=None)

    def test_insert_after_full(self):
        memory = ReplayMemory(
            capacity=self.capacity,
            next_states=True
        )
        test = ComponentTest(component=memory, input_spaces=dict(
            records=self.record_space,
            num_records=int
        ))
        buffer_size, buffer_index = memory.get_variables()

        size_value, index_value = test.get_variable_values([buffer_size, buffer_index])
        # Assert indices 0 before insert.
        self.assertTrue(size_value == 0)
        self.assertTrue(index_value == 0)

        # Insert one more element than capacity
        observation = self.record_space.sample(size=self.capacity + 1)
        test.test(out_socket_name="insert", inputs=observation, expected_outputs=None)

        size_value, index_value = test.get_variable_values([buffer_size, buffer_index])
        # Size should be full now
        self.assertTrue(size_value == self.capacity)
        # One over capacity
        self.assertTrue(index_value == 1)

    def test_batch_retrieve(self):
        memory = ReplayMemory(
            capacity=self.capacity,
            next_states=True
        )
        test = ComponentTest(component=memory, input_spaces=dict(records=self.record_space))
        buffer_size, buffer_index = memory.get_variables()

        # Assert nothing in here yet.
        num_records = 1
        batch = test.test(out_socket_name="get_records", inputs=num_records, expected_outputs=None)
        self.assertEqual(0, len(batch['terminal']))

        # Insert 2 Elements.
        observation = self.record_space.sample(size=2)
        test.test(out_socket_name="insert", inputs=observation, expected_outputs=None)

        # Assert we can now fetch 2 elements.
        num_records = 2
        batch = test.test(out_socket_name="get_records", inputs=num_records, expected_outputs=None)
        self.assertEqual(2, len(batch['terminal']))

        # Assert we cannot fetch more than 2 elements because size is 2.
        num_records = 5
        batch = test.test(out_socket_name="get_records", inputs=num_records, expected_outputs=None)
        self.assertEqual(2, len(batch['terminal']))

        # Now insert over capacity.
        observation = self.record_space.sample(size=self.capacity)
        test.test(out_socket_name="insert", inputs=observation, expected_outputs=None)

        # Assert we can fetch exactly capacity elements.
        num_records = self.capacity
        batch = test.test(out_socket_name="get_records", inputs=num_records, expected_outputs=None)
        self.assertEqual(self.capacity, len(batch['terminal']))

    def test_episode_semantics(self):
        # TODO
        pass

    def test_latest_semantics(self):
        # TODO
        pass
