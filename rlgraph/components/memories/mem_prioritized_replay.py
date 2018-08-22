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

from rlgraph.utils.specifiable import Specifiable
from rlgraph.components.helpers.mem_segment_tree import MemSegmentTree, MinSumSegmentTree
from rlgraph.spaces.space_utils import get_list_registry
from rlgraph.spaces import Dict


class MemPrioritizedReplay(Specifiable):
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

    # TODO this needs manual calling atm
    def create_variables(self, input_spaces, action_space=None):
        # Store our record-space for convenience.
        self.record_space = input_spaces["records"]
        self.record_space_flat = Dict(self.record_space.flatten(custom_scope_separator="-",
                                                                scope_separator_at_start=False),
                                      add_batch_rank=True)

        # Create the main memory as a flattened OrderedDict from any arbitrarily nested Space.
        self.record_registry = get_list_registry(self.record_space_flat)
        self.fixed_key = list(self.record_registry.keys())[0]

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

        if self.next_states:
            assert 'states' in self.record_space
            # Next states are not represented as explicit keys in the registry
            # as this would cause extra memory overhead.
            self.flat_state_keys = [k for k in self.record_registry.keys() if k[:7] == "states-"]

    def insert_records(self, records):
        num_records = len(records[self.fixed_key])

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
                record = dict()
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

    # def insert_records(self, records):
    #     num_records = len(records[self.fixed_key])
    #     update_indices = np.arange(start=self.index, stop=self.index + num_records) % self.capacity
    #
    #     # Update record registry.
    #     if num_records == 1:
    #         # TODO no indices exist so we have to append or presize
    #         insert_index = (self.index + num_records) % self.capacity
    #         for key in self.record_registry:
    #             self.record_registry[key][insert_index] = records[key]
    #     else:
    #         insert_indices = np.arange(start=self.index, stop=self.index + num_records) % self.capacity
    #         record_index = 0
    #         for insert_index in insert_indices:
    #             for key in self.record_registry:
    #                 self.record_registry[key][insert_index] = records[key][record_index]
    #             record_index += 1
    #
    #     # Update indices
    #     self.index = (self.index + num_records) % self.capacity
    #     self.size = min(self.size + num_records, self.capacity)
    #
    #     # Insert into segment trees.
    #     for i in range_(num_records):
    #         self.sum_segment_tree.insert(update_indices[i], self.default_new_weight)
    #         self.min_segment_tree.insert(update_indices[i], self.default_new_weight)

    # def read_records(self, indices):
    #     """
    #     Obtains record values for the provided indices.
    #
    #     Args:
    #         indices ndarray: Indices to read. Assumed to be not contiguous.
    #
    #     Returns:
    #          dict: Record value dict.
    #     """
    #     records = dict()
    #     for name, variable in self.record_registry.items():
    #             records[name] = [variable[index] for index in indices]
    #     if self.next_states:
    #         next_indices = (indices + 1) % self.capacity
    #
    #         # Next states are read via index shift from state variables.
    #         for flat_state_key in self.flat_state_keys:
    #             next_states = [self.record_registry[flat_state_key][index] for index in next_indices]
    #             flat_next_state_key = "next_states"+flat_state_key[len("states"):]
    #             records[flat_next_state_key] = next_states
    #     return records

    def read_records(self, indices):
        """
        Obtains record values for the provided indices.

        Args:
            indices ndarray: Indices to read. Assumed to be not contiguous.

        Returns:
             dict: Record value dict.
        """
        records = dict()
        for name in self.record_registry.keys():
            records[name] = []
        if self.next_states:
            for flat_state_key in self.flat_state_keys:
                flat_next_state_key = "next_states" + flat_state_key[len("states"):]
                records[flat_next_state_key] = []

        for index in indices:
            record = self.memory_values[index]
            for name in self.record_registry.keys():
                records[name].append(record[name])

            if self.next_states:
                # TODO these are largely copies
                next_index = (index + 1) % self.capacity
                for flat_state_key in self.flat_state_keys:
                    next_record = self.memory_values[next_index]
                    flat_next_state_key = "next_states"+flat_state_key[len("states"):]
                    records[flat_next_state_key].append(next_record[flat_state_key])
        return records

    def get_records(self, num_records):
        indices = []
        prob_sum = self.merged_segment_tree.sum_segment_tree.get_sum(0, self.size - 1)
        samples = np.random.random(size=(num_records,)) * prob_sum
        for sample in samples:
            indices.append(self.merged_segment_tree.sum_segment_tree.index_of_prefixsum(prefix_sum=sample))

        sum_prob = self.merged_segment_tree.sum_segment_tree.get_sum()
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

