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
from six.moves import xrange as range_
from rlgraph.components.memories.mem_prioritized_replay import MemPrioritizedReplay
from rlgraph.execution.ray.apex.apex_memory import ApexMemory
from rlgraph.execution.ray.ray_util import ray_compress
from rlgraph.spaces import Dict, IntBox, BoolBox, FloatBox


class TestPythonPrioritizedReplay(unittest.TestCase):
    """
    Tests sampling and insertion behaviour of the mem_prioritized_replay module.
    """
    record_space = Dict(
        states=dict(state1=float, state2=float),
        actions=dict(action1=float),
        reward=float,
        terminals=BoolBox(),
        add_batch_rank=True
    )
    apex_space = Dict(
        states=FloatBox(shape=(4,)),
        actions=FloatBox(shape=(2,)),
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
        insert_records=[record_space],
        # get_records: num_records
        get_records=[int],
        # update_records: indices, update
        update_records=[IntBox(shape=(), add_batch_rank=True), FloatBox(shape=(), add_batch_rank=True)]
    )

    def test_insert(self):
        """
        Simply tests insert op without checking internal logic.
        """
        memory = MemPrioritizedReplay(
            capacity=self.capacity,
            next_states=True,
            alpha=self.alpha,
            beta=self.beta
        )
        memory.create_variables(self.input_spaces, None)

        observation = memory.record_space_flat.sample(size=1)
        memory.insert_records(observation)

        # Test chunked insert
        observation = memory.record_space_flat.sample(size=5)
        memory.insert_records(observation)

        # Also test Apex version
        memory = ApexMemory(
            capacity=self.capacity,
            alpha=self.alpha,
            beta=self.beta
        )
        observation = self.apex_space.sample(size=5)
        for i in range_(5):
            memory.insert_records((
                observation['states'][i],
                observation['actions'][i],
                observation['reward'][i],
                observation['terminals'][i]
            ))

    def test_update_records(self):
        """
        Tests update records logic.
        """
        memory = MemPrioritizedReplay(
            capacity=self.capacity,
            next_states=True
        )
        memory.create_variables(self.input_spaces, None)

        # Insert a few Elements.
        observation = memory.record_space_flat.sample(size=2)
        memory.insert_records(observation)

        # Fetch elements and their indices.
        num_records = 2
        batch = memory.get_records(num_records)
        indices = batch[1]
        self.assertEqual(num_records, len(indices))

        # Does not return anything.
        memory.update_records(indices, np.asarray([0.1, 0.2]))

        # Test apex memory.
        memory = ApexMemory(
            capacity=self.capacity,
            alpha=self.alpha,
            beta=self.beta
        )
        observation = self.apex_space.sample(size=5)
        for i in range_(5):
            memory.insert_records((
                ray_compress(observation['states'][i]),
                observation['actions'][i],
                observation['reward'][i],
                observation['terminals'][i]
            ))

        # Fetch elements and their indices.
        num_records = 5
        batch = memory.get_records(num_records)
        indices = batch[1]
        self.assertEqual(num_records, len(indices))

        # Does not return anything
        memory.update_records(indices, np.random.uniform(size=10))

    def test_segment_tree_insert_values(self):
        """
        Tests if segment tree inserts into correct positions.
        """
        memory = MemPrioritizedReplay(
            capacity=self.capacity,
            next_states=True,
            alpha=self.alpha,
            beta=self.beta
        )
        memory.create_variables(self.input_spaces, None)

        priority_capacity = 1
        while priority_capacity < self.capacity:
            priority_capacity *= 2

        sum_segment_values = memory.merged_segment_tree.sum_segment_tree.values
        min_segment_values = memory.merged_segment_tree.min_segment_tree.values

        self.assertEqual(sum(sum_segment_values), 0)
        self.assertEqual(sum(min_segment_values), float('inf'))
        self.assertEqual(len(sum_segment_values), 2 * priority_capacity)
        self.assertEqual(len(min_segment_values), 2 * priority_capacity)

        # Insert 1 Element.
        observation = memory.record_space_flat.sample(size=1)
        memory.insert_records(observation)

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
        observation =  memory.record_space_flat.sample(size=1)
        memory.insert_records(observation)

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