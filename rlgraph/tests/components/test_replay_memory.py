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

import unittest

from rlgraph.components.memories.replay_memory import ReplayMemory
from rlgraph.spaces import Dict, BoolBox
from rlgraph.tests import ComponentTest
from rlgraph.tests.test_util import non_terminal_records, terminal_records


class TestReplayMemory(unittest.TestCase):
    """
    Tests sampling and insertion behaviour of the replay_memory module.
    """
    record_space = Dict(
        states=dict(state1=float, state2=float),
        actions=dict(action1=float),
        reward=float,
        terminals=BoolBox(),
        next_states=dict(state1=float, state2=float),
        add_batch_rank=True
    )
    memory_variables = ["size", "index"]
    capacity = 10

    input_spaces = dict(
        records=record_space,
        num_records=int
    )

    def test_insert(self):
        """
        Simply tests insert op without checking internal logic.
        """
        memory = ReplayMemory(
            capacity=self.capacity,
        )
        test = ComponentTest(component=memory, input_spaces=self.input_spaces)

        observation = self.record_space.sample(size=1)
        test.test(("insert_records", observation), expected_outputs=None)

        observation = self.record_space.sample(size=100)
        test.test(("insert_records", observation), expected_outputs=None)

    def test_capacity(self):
        """
        Tests if insert correctly manages capacity.
        """
        memory = ReplayMemory(
            capacity=self.capacity
        )
        test = ComponentTest(component=memory, input_spaces=self.input_spaces)
        # Internal state variables.
        memory_variables = memory.get_variables(self.memory_variables, global_scope=False)
        buffer_size = memory_variables['size']
        buffer_index = memory_variables['index']
        size_value, index_value = test.read_variable_values(buffer_size, buffer_index)

        # Assert indices 0 before insert.
        self.assertEqual(size_value, 0)
        self.assertEqual(index_value, 0)

        # Insert one more element than capacity
        observation = self.record_space.sample(size=self.capacity + 1)
        test.test(("insert_records", observation), expected_outputs=None)

        size_value, index_value = test.read_variable_values(buffer_size, buffer_index)
        # Size should be equivalent to capacity when full.
        self.assertEqual(size_value, self.capacity)

        # Index should be one over capacity due to modulo.
        self.assertEqual(index_value, 1)

    def test_batch_retrieve(self):
        """
        Tests if retrieval correctly manages capacity.
        """
        memory = ReplayMemory(
            capacity=self.capacity
        )
        test = ComponentTest(component=memory, input_spaces=self.input_spaces)

        # Insert 2 Elements.
        observation = non_terminal_records(self.record_space, 2)
        test.test(("insert_records", observation), expected_outputs=None)

        # Assert we can now fetch 2 elements.
        num_records = 2
        batch, _, _ = test.test(("get_records", num_records), expected_outputs=None)
        print('Result batch = {}'.format(batch))
        self.assertEqual(2, len(batch['terminals']))
        # Assert next states key is there
        self.assertTrue('next_states' in batch)

        # Test duplicate sampling.
        num_records = 5
        batch, _, _ = test.test(("get_records", num_records), expected_outputs=None)
        self.assertEqual(5, len(batch['terminals']))

        # Now insert over capacity.
        observation = non_terminal_records(self.record_space, self.capacity)
        test.test(("insert_records", observation), expected_outputs=None)

        # Assert we can fetch exactly capacity elements.
        num_records = self.capacity
        batch, _, _ = test.test(("get_records", num_records), expected_outputs=None)
        self.assertEqual(self.capacity, len(batch['terminals']))
