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
import numpy as np

from rlgraph.components.memories import PrioritizedReplay
from rlgraph.spaces import Dict, IntBox, BoolBox, FloatBox
from rlgraph.tests import ComponentTest
from rlgraph.tests.test_util import non_terminal_records


class TestPrioritizedReplay(unittest.TestCase):
    """
    Tests sampling and insertion behaviour of the prioritized_replay module.
    """
    record_space = Dict(
        states=dict(state1=float, state2=float),
        actions=dict(action1=float),
        reward=float,
        terminals=BoolBox(),
        add_batch_rank=True
    )
    memory_variables = ["size", "index", "max-priority"]

    capacity = 10
    alpha = 1.0
    beta = 1.0

    max_priority = 1.0

    input_spaces = dict(
        # insert: records
        records=record_space,
        # get_records: num_records
        num_records=int,
        # update_records: indices, update
        indices=IntBox(add_batch_rank=True),
        update=FloatBox(add_batch_rank=True)
    )

    def test_insert(self):
        """
        Simply tests insert op without checking internal logic.
        """
        memory = PrioritizedReplay(
            capacity=self.capacity,
            alpha=self.alpha,
            beta=self.beta
        )
        test = ComponentTest(component=memory, input_spaces=self.input_spaces)

        observation = self.record_space.sample(size=1)
        test.test(("insert_records", observation), expected_outputs=None)

    def test_capacity(self):
        """
        Tests if insert correctly manages capacity.
        """
        memory = PrioritizedReplay(
            capacity=self.capacity,
            alpha=self.alpha,
            beta=self.beta
        )
        test = ComponentTest(component=memory, input_spaces=self.input_spaces)

        # Internal state variables.
        memory_variables = memory.get_variables(self.memory_variables, global_scope=False)
        buffer_size = memory_variables['size']
        buffer_index = memory_variables['index']
        max_priority = memory_variables['max-priority']

        size_value, index_value, max_priority_value = test.read_variable_values(buffer_size, buffer_index, max_priority)

        # Assert indices 0 before insert.
        self.assertEqual(size_value, 0)
        self.assertEqual(index_value, 0)
        self.assertEqual(max_priority_value, 1.0)

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
        memory = PrioritizedReplay(
            capacity=self.capacity,
            alpha=self.alpha,
            beta=self.beta
        )
        test = ComponentTest(component=memory, input_spaces=self.input_spaces)

        # Insert 2 Elements.
        observation = non_terminal_records(self.record_space, 2)
        test.test(("insert_records", observation), expected_outputs=None)

        # Assert we can now fetch 2 elements.
        num_records = 2
        batch = test.test(("get_records", num_records), expected_outputs=None)
        records = batch[0]
        print('Result batch = {}'.format(records))
        self.assertEqual(2, len(records['terminals']))

        # We allow repeat indices in sampling.
        num_records = 5
        batch = test.test(("get_records", num_records), expected_outputs=None)
        records = batch[0]
        self.assertEqual(5, len(records['terminals']))

        # Now insert over capacity, note all elements here are non-terminal.
        observation = non_terminal_records(self.record_space, self.capacity)
        test.test(("insert_records", observation), expected_outputs=None)

        # Assert we can fetch exactly capacity elements.
        num_records = self.capacity
        batch = test.test(("get_records", num_records), expected_outputs=None)
        records = batch[0]
        self.assertEqual(self.capacity, len(records['terminals']))

    def test_update_records(self):
        """
        Tests update records logic.
        """
        memory = PrioritizedReplay(
            capacity=self.capacity
        )
        test = ComponentTest(component=memory, input_spaces=self.input_spaces)

        # Insert a few Elements.
        observation = non_terminal_records(self.record_space, 2)
        test.test(("insert_records", observation), expected_outputs=None)

        # Fetch elements and their indices.
        num_records = 2
        batch = test.test(("get_records", num_records), expected_outputs=None)
        indices = batch[1]
        self.assertEqual(num_records, len(indices))
        # 0.3, 0.5, 1.0])
        input_params = [indices, np.asarray([0.1, 0.2])]
        # Does not return anything
        test.test(("update_records", input_params), expected_outputs=None)

    def test_segment_tree_insert_values(self):
        """
        Tests if segment tree inserts into correct positions.
        """
        memory = PrioritizedReplay(
            capacity=self.capacity,
            alpha=self.alpha,
            beta=self.beta
        )
        test = ComponentTest(component=memory, input_spaces=self.input_spaces)
        priority_capacity = 1
        while priority_capacity < self.capacity:
            priority_capacity *= 2

        memory_variables = memory.get_variables(["sum-segment-tree", "min-segment-tree"], global_scope=False)
        sum_segment_tree = memory_variables['sum-segment-tree']
        min_segment_tree = memory_variables['min-segment-tree']
        sum_segment_values, min_segment_values = test.read_variable_values(sum_segment_tree, min_segment_tree)

        self.assertEqual(sum(sum_segment_values), 0)
        self.assertEqual(sum(min_segment_values), float('inf'))
        self.assertEqual(len(sum_segment_values), 2 * priority_capacity)
        self.assertEqual(len(min_segment_values), 2 * priority_capacity)
        # Insert 1 Element.
        observation = non_terminal_records(self.record_space, 1)
        test.test(("insert_records", observation), expected_outputs=None)

        # Fetch segment tree.
        sum_segment_values, min_segment_values = test.read_variable_values(sum_segment_tree, min_segment_tree)

        # Check insert positions
        # Initial insert is at priority capacity
        print(sum_segment_values)
        print(min_segment_values)
        start = priority_capacity

        while start >= 1:
            self.assertEqual(sum_segment_values[start], 1.0)
            self.assertEqual(min_segment_values[start], 1.0)
            start = int(start / 2)

        # Insert another Element.
        observation = non_terminal_records(self.record_space, 1)
        test.test(("insert_records", observation), expected_outputs=None)

        # Fetch segment tree.
        sum_segment_values, min_segment_values = test.read_variable_values(sum_segment_tree, min_segment_tree)
        print(sum_segment_values)
        print(min_segment_values)

        # Index shifted 1
        start = priority_capacity + 1
        self.assertEqual(sum_segment_values[start], 1.0)
        self.assertEqual(min_segment_values[start], 1.0)
        start = int(start / 2)
        while start >= 1:
            # 1 + 1 is 2 on the segment.
            self.assertEqual(sum_segment_values[start], 2.0)
            # min is still 1.
            self.assertEqual(min_segment_values[start], 1.0)
            start = int(start / 2)