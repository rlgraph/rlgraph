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

from yarl.components.memories.ring_buffer import RingBuffer
from yarl.spaces import Dict, IntBox
from yarl.tests import ComponentTest
from yarl.tests.test_util import non_terminal_records, terminal_records


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

    def test_insert_no_episodes(self):
        """
        Simply tests insert op without checking internal logic, episode
        semantics disabled.
        """
        ring_buffer = RingBuffer(capacity=self.capacity, episode_semantics=False)
        test = ComponentTest(component=ring_buffer, input_spaces=dict(
            records=self.record_space,
            num_records=int
        ))

        observation = self.record_space.sample(size=1)
        test.test(out_socket_name="insert", inputs=observation, expected_outputs=None)

        observation = self.record_space.sample(size=100)
        test.test(out_socket_name="insert", inputs=observation, expected_outputs=None)

    def test_capacity_no_episodes(self):
        """
        Tests if insert correctly manages capacity, no episode indices updated..
        """
        ring_buffer = RingBuffer(capacity=self.capacity, episode_semantics=False)
        test = ComponentTest(component=ring_buffer, input_spaces=dict(
            records=self.record_space,
            num_records=int
        ))
        # Internal state variables.
        buffer_size, buffer_index = ring_buffer.get_variables()
        size_value, index_value = test.get_variable_values([buffer_size, buffer_index])

        # Assert indices 0 before insert.
        self.assertEqual(size_value, 0)
        self.assertEqual(index_value, 0)

        # Insert one more element than capacity
        observation = self.record_space.sample(size=self.capacity + 1)
        test.test(out_socket_name="insert", inputs=observation, expected_outputs=None)

        size_value, index_value = test.get_variable_values([buffer_size, buffer_index])
        # Size should be equivalent to capacity when full.
        self.assertEqual(size_value, self.capacity)

        # Index should be one over capacity due to modulo.
        self.assertEqual(index_value, 1)

    def test_capacity_with_episodes(self):
        """
        Tests if episode variable updates work when meeting capacity.

        Note that this does not test episode semantics itself, which are tested below.
        """
        ring_buffer = RingBuffer(capacity=self.capacity, episode_semantics=True)
        test = ComponentTest(component=ring_buffer, input_spaces=dict(
            records=self.record_space,
            num_records=int
        ))
        # Internal state variables.
        buffer_size, buffer_index, num_episodes, episode_indices = ring_buffer.get_variables()
        size_value, index_value, num_episodes_value = test.get_variable_values([buffer_size, buffer_index, num_episodes])

        # Assert indices 0 before insert.
        self.assertEqual(size_value, 0)
        self.assertEqual(index_value, 0)
        self.assertEqual(num_episodes_value, 0)

        # Insert one more element than capacity. Note: this is different than
        # replay test because due to episode semantics, it matters if
        # these are terminal or not. This tests if episode index updating
        # causes problems if none of the inserted elements are terminal.
        observation = non_terminal_records(self.record_space, self.capacity + 1)
        test.test(out_socket_name="insert", inputs=observation, expected_outputs=None)
        size_value, index_value, num_episodes_value = test.get_variable_values([buffer_size, buffer_index, num_episodes])

        # Size should be equivalent to capacity when full.
        self.assertEqual(size_value, self.capacity)

        # Index should be one over capacity due to modulo.
        self.assertEqual(index_value, 1)
        self.assertEqual(num_episodes_value, 0)

        # # Next, we insert a single terminal record.
        # observation = terminal_records(self.record_space, 1)
        # test.test(out_socket_name="insert", inputs=observation, expected_outputs=None)
        # num_episodes, episode_indices = test.get_variable_values([num_episodes, episode_indices])

    def test_episode_semantics(self):
        # TODO
        pass

    def test_latest_semantics(self):
        # TODO
        pass
