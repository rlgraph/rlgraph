# Copyright 2018/2019 The RLgraph authors. All Rights Reserved.
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

from rlgraph import get_backend
from rlgraph.utils import util, DataOpDict
from rlgraph.utils.define_by_run_ops import define_by_run_unflatten
from rlgraph.utils.util import SMALL_NUMBER, get_rank
from rlgraph.components.memories.memory import Memory
from rlgraph.components.helpers.mem_segment_tree import MemSegmentTree, MinSumSegmentTree
from rlgraph.utils.decorators import rlgraph_api

if get_backend() == "pytorch":
    import torch


class MemPrioritizedReplay(Memory):
    """
    Implements an in-memory  prioritized replay.

    API:
        update_records(indices, update) -> Updates the given indices with the given priority scores.
    """
    def __init__(self, capacity=1000, next_states=True, alpha=1.0, beta=0.0):
        super(MemPrioritizedReplay, self).__init__()

        self.memory_values = []
        self.index = 0
        self.capacity = capacity

        self.size = 0
        self.max_priority = 1.0
        self.alpha = alpha
        self.beta = beta
        self.next_states = next_states

        self.default_new_weight = np.power(self.max_priority, self.alpha)

    def create_variables(self, input_spaces, action_space=None):
        super(MemPrioritizedReplay, self).create_variables(input_spaces, action_space)
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

    @rlgraph_api(flatten_ops=True)
    def _graph_fn_insert_records(self, records):
        if records is None or get_rank(records[self.terminal_key]) == 0:
            return
        num_records = len(records[self.terminal_key])

        if num_records == 1:
            if self.index >= self.size:
                self.memory_values.append(records)
            else:
                self.memory_values[self.index] = records
            self.merged_segment_tree.insert(self.index, self.default_new_weight)
        else:
            insert_indices = np.arange(start=self.index, stop=self.index + num_records) % self.capacity
            i = 0
            for insert_index in insert_indices:
                self.merged_segment_tree.insert(insert_index, self.default_new_weight)
                record = {}
                for name, record_values in records.items():
                    record[name] = record_values[i]
                if insert_index >= self.size:
                    self.memory_values.append(record)
                else:
                    self.memory_values[insert_index] = record
                i += 1

        # Update indices
        self.index = (self.index + num_records) % self.capacity
        self.size = min(self.size + num_records, self.capacity)

    @rlgraph_api
    def _graph_fn_get_records(self, num_records=1):
        available_records = min(num_records, self.size)
        indices = []
        prob_sum = self.merged_segment_tree.sum_segment_tree.get_sum(0, self.size - 1)
        samples = np.random.random(size=(available_records,)) * prob_sum
        for sample in samples:
            indices.append(self.merged_segment_tree.sum_segment_tree.index_of_prefixsum(prefix_sum=sample))

        sum_prob = self.merged_segment_tree.sum_segment_tree.get_sum() + SMALL_NUMBER
        min_prob = self.merged_segment_tree.min_segment_tree.get_min_value() / sum_prob
        max_weight = (min_prob * self.size) ** (-self.beta)
        weights = []
        for index in indices:
            sample_prob = self.merged_segment_tree.sum_segment_tree.get(index) / sum_prob
            weight = (sample_prob * self.size) ** (-self.beta)
            weights.append(weight / max_weight)

        if get_backend() == "pytorch":
            indices = torch.tensor(indices)
            weights = torch.tensor(weights)
        else:
            indices = np.asarray(indices)
            weights = np.asarray(weights)

        records = DataOpDict()
        for name, variable in self.memory.items():
            records[name] = self.read_variable(variable, indices, dtype=
            util.convert_dtype(self.flat_record_space[name].dtype, to="pytorch"))
        records = define_by_run_unflatten(records)
        return records, indices, weights

    @rlgraph_api(must_be_complete=False)
    def _graph_fn_update_records(self, indices, update):
        for index, loss in zip(indices, update):
            priority = np.power(loss, self.alpha)
            self.merged_segment_tree.insert(index, priority)
            self.max_priority = max(self.max_priority, priority)

    def get_state(self):
        return {
            "size": self.size,
            "index": self.index,
            "max_priority": self.max_priority
        }