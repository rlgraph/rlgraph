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

import numpy as np
import operator
from six.moves import xrange as range_

from rlgraph import Specifiable
from rlgraph.components.memories.mem_segment_tree import MemSegmentTree, MinSumSegmentTree
from rlgraph.execution.ray.ray_util import ray_decompress


class ApexMemory(Specifiable):
    """
    Apex prioritized replay implementing compression.
    """
    def __init__(self, capacity=1000, alpha=1.0, beta=0.0, n_step=1):
        self.memory_values = []
        self.index = 0
        self.capacity = capacity
        self.size = 0
        self.max_priority = 1.0
        self.alpha = alpha
        self.beta = beta
        self.nstep = n_step

        self.default_new_weight = np.power(self.max_priority, self.alpha)
        self.priority_capacity = 1
        while self.priority_capacity < self.capacity:
            self.priority_capacity *= 2

        # Create segment trees, initialize with neutral elements.
        sum_values = [0.0 for _ in range_(2 * self.priority_capacity)]
        sum_segment_tree = MemSegmentTree(sum_values, self.priority_capacity, operator.add)
        min_values = [float('inf') for _ in range_(2 * self.priority_capacity)]
        min_segment_tree = MemSegmentTree(min_values, self.priority_capacity, min)
        self.merged_segment_tree = MinSumSegmentTree(
            sum_tree=sum_segment_tree,
            min_tree=min_segment_tree,
            capacity=self.priority_capacity
        )

    def insert_records(self, record):
        # TODO: This has the record interface, but actually expects a specific structure anyway, so
        # may as well change API?
        if self.index >= self.size:
            self.memory_values.append(record)
        else:
            self.memory_values[self.index] = record

        # Weights.
        if record[4] is not None:
            self.merged_segment_tree.insert(self.index, record[4])
        else:
            self.merged_segment_tree.insert(self.index, self.default_new_weight)

        # Update indices.
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def read_records(self, indices):
        """
        Obtains record values for the provided indices.

        Args:
            indices ndarray: Indices to read. Assumed to be not contiguous.

        Returns:
             dict: Record value dict.
        """
        states = []
        actions = []
        rewards = []
        terminals = []
        next_states = []
        for index in indices:
            record = self.memory_values[index]
            state, action, reward, terminal, weight = record
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            terminals.append(terminal)
            next_index = (index + self.nstep) % self.capacity

            next_state = self.memory_values[next_index][0]
            next_states.append(next_state)

        return dict(
            states=[ray_decompress(state) for state in states],
            actions=actions,
            rewards=rewards,
            terminals=terminals,
            next_states=[ray_decompress(state) for state in next_states]
        )

    def get_records(self, num_records):
        indices = []
        prob_sum = self.merged_segment_tree.sum_segment_tree.get_sum(0, self.size - 1)
        samples = np.random.random(size=(num_records,)) * prob_sum
        for sample in samples:
            indices.append(self.merged_segment_tree.sum_segment_tree.index_of_prefixsum(prefix_sum=sample))

        sum_prob = self.merged_segment_tree.sum_segment_tree.reduce(start=0, limit=self.priority_capacity - 1)
        min_prob = self.merged_segment_tree.min_segment_tree.get_min_value() / sum_prob
        max_weight = (min_prob * self.size) ** (-self.beta)
        weights = []
        for index in indices:
            sample_prob = self.merged_segment_tree.sum_segment_tree.get(index) / sum_prob
            weight = (sample_prob * self.size) ** (-self.beta)
            weights.append(weight / max_weight)

        indices = np.asarray(indices)
        return self.read_records(indices=indices), indices, np.asarray(weights)

    def update_records(self, indices, update):
        for index, loss in zip(indices, update):
            priority = np.power(loss, self.alpha)
            self.merged_segment_tree.insert(index, priority)
            self.max_priority = max(self.max_priority, priority)

