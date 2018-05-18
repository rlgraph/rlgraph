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
from yarl.tests.test_util import non_terminal_records, terminal_records


class TestReplayMemory(unittest.TestCase):
    """
    Tests sampling and insertion behaviour of the replay_memory module.
    """
    record_space = Dict(
        states=dict(state1=float, state2=float),
        actions=dict(action1=float),
        reward=float,
        terminal=IntBox(low=0, high=1),
        add_batch_rank=True
    )
    capacity = 10

    def test_insert(self):
        """
        Simply tests insert op without checking internal logic.
        """
        memory = ReplayMemory(
            capacity=self.capacity,
            next_states=True
        )
        test = ComponentTest(component=memory, input_spaces=dict(
            records=self.record_space,
            num_records=int
        ))

        observation = self.record_space.sample(size=1)
        test.test(out_socket_name="insert", inputs=observation, expected_outputs=None)

        observation = self.record_space.sample(size=100)
        test.test(out_socket_name="insert", inputs=observation, expected_outputs=None)

    def test_capacity(self):
        """
        Tests if insert correctly manages capacity.
        """
        memory = ReplayMemory(
            capacity=self.capacity,
            next_states=True
        )
        test = ComponentTest(component=memory, input_spaces=dict(
            records=self.record_space,
            num_records=int
        ))
        # Internal state variables.
        buffer_size, buffer_index = memory.get_variables()
        size_value, index_value = test.get_variable_values([buffer_size, buffer_index])

        # Assert indices 0 before insert.
        self.assertTrue(size_value == 0)
        self.assertTrue(index_value == 0)

        # Insert one more element than capacity
        observation = self.record_space.sample(size=self.capacity + 1)
        test.test(out_socket_name="insert", inputs=observation, expected_outputs=None)

        size_value, index_value = test.get_variable_values([buffer_size, buffer_index])
        # Size should be equivalent to capacity when full.
        self.assertTrue(size_value == self.capacity)

        # Index should be one over capacity due to modulo.
        self.assertTrue(index_value == 1)

    def test_batch_retrieve(self):
        """
        Tests if retrieval correctly manages capacity and zeros out terminals.
        """
        memory = ReplayMemory(
            capacity=self.capacity,
            next_states=True
        )
        test = ComponentTest(component=memory, input_spaces=dict(
            records=self.record_space,
            num_records=int
        ))

        # Insert 2 Elements.
        observation = non_terminal_records(self.record_space, 2)
        test.test(out_socket_name="insert", inputs=observation, expected_outputs=None)

        # Assert we can now fetch 2 elements.
        num_records = 2
        batch = test.test(out_socket_name="sample", inputs=num_records, expected_outputs=None)
        print('Result batch = {}'.format(batch))
        self.assertEqual(2, len(batch['terminal']))

        # Assert we cannot fetch more than 2 elements because size is 2.
        num_records = 5
        batch = test.test(out_socket_name="sample", inputs=num_records, expected_outputs=None)
        self.assertEqual(2, len(batch['terminal']))

        # Now insert over capacity, note all elements here are non-terminal.
        observation = non_terminal_records(self.record_space, self.capacity)
        test.test(out_socket_name="insert", inputs=observation, expected_outputs=None)

        # Assert we can fetch exactly capacity elements.
        num_records = self.capacity
        batch = test.test(out_socket_name="sample", inputs=num_records, expected_outputs=None)
        self.assertEqual(self.capacity, len(batch['terminal']))

        # Now insert 5 terminal elements.
        observation = terminal_records(self.record_space, 5)
        test.test(out_socket_name="insert", inputs=observation, expected_outputs=None)

        # Try to fetch capacity elements again.
        num_records = self.capacity
        batch = test.test(out_socket_name="sample", inputs=num_records, expected_outputs=None)

        # We now expect to be able to sample capacity - 5 elements due to terminals.
        expected = self.capacity - 5
        self.assertEqual(expected, len(batch['terminal']))
