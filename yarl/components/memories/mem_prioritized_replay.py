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

import numpy as np
import operator
from six.moves import xrange

from yarl import Specifiable
from yarl.components.memories.mem_segment_tree import MemSegmentTree
from yarl.spaces.space_utils import get_list_registry
from yarl.utils.ops import FlattenedDataOp
from yarl.utils.util import get_batch_size


class MemPrioritizedReplay(Specifiable):
    """
    Implements an in-memory  prioritized replay.

    API:
        update_records(indices, update) -> Updates the given indices with the given priority scores.
    """
    def __init__(self, capacity=1000, next_states=True, alpha=1.0, beta=0.0):
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
    def create_variables(self, input_spaces, action_space):
        # Store our record-space for convenience.
        self.record_space = input_spaces["insert_records"][0]

        # Create the main memory as a flattened OrderedDict from any arbitrarily nested Space.
        self.record_registry = get_list_registry(self.record_space)

        self.priority_capacity = 1
        while self.priority_capacity < self.capacity:
            self.priority_capacity *= 2

        # Create segment trees, initialize with neutral elements.
        sum_values = [0.0 for _ in xrange(2 * self.priority_capacity)]
        self.sum_segment_tree = MemSegmentTree(sum_values, self.priority_capacity, operator.add)
        min_values = [float('inf') for _ in xrange(2 * self.priority_capacity)]
        self.min_segment_tree = MemSegmentTree(min_values, self.priority_capacity, min)

    def insert_records(self, records):
        num_records = len(records["/terminals"])
        update_indices = np.arange(start=self.index, stop=self.index + num_records) % self.capacity

        # Update record registry.
        if num_records == 1:
            insert_index = (self.index + num_records) % self.capacity
            for key in self.record_registry:
                self.record_registry[key][insert_index] = records[key]
        else:
            insert_indices = np.arange(start=self.index, stop=self.index + num_records) % self.capacity
            record_index = 0
            for insert_index in insert_indices:
                for key in self.record_registry:
                    self.record_registry[key][insert_index] = records[key][record_index]
                record_index += 1

        # Update indices
        self.index = (self.index + num_records) % self.capacity
        self.size = min(self.size + num_records, self.capacity)

        # Insert into segment trees.
        for i in xrange(num_records):
            self.sum_segment_tree.insert(update_indices[i], self.default_new_weight)
            self.min_segment_tree.insert(update_indices[i], self.default_new_weight)

    def get_records(self, num_records):
        pass

        # current_size = self.read_variable(self.size)
        # stored_elements_prob_sum = self.sum_segment_tree.reduce(start=0, limit=current_size - 1)
        #
        # # Sample the entire batch.
        # sample = stored_elements_prob_sum * tf.random_uniform(shape=(num_records,))
        #
        # # Sample by looking up prefix sum.
        # sample_indices = tf.map_fn(fn=self.sum_segment_tree.index_of_prefixsum, elems=sample, dtype=tf.int32)
        # # sample_indices = self.sum_segment_tree.index_of_prefixsum(sample)
        #
        # # Importance correction.
        # total_prob = self.sum_segment_tree.reduce(start=0, limit=self.priority_capacity - 1)
        # min_prob = self.min_segment_tree.get_min_value() / total_prob
        # max_weight = tf.pow(x=min_prob * tf.cast(current_size, tf.float32), y=-self.beta)
        #
        # def importance_sampling_fn(sample_index):
        #     sample_prob = self.sum_segment_tree.get(sample_index) / stored_elements_prob_sum
        #     weight = tf.pow(x=sample_prob * tf.cast(current_size, tf.float32), y=-self.beta)
        #
        #     return weight / max_weight
        #
        # # sample_indices = tf.Print(sample_indices, [sample_indices], summarize=1000,
        # #                          message='sample indices in retrieve = ')
        #
        # corrected_weights = tf.map_fn(
        #     fn=importance_sampling_fn,
        #     elems=sample_indices,
        #     dtype=tf.float32
        # )
        # sample_indices = tf.Print(sample_indices, [sample_indices, self.sum_segment_tree.values], summarize=1000,
        #                           message='sample indices, segment tree values = ')
        # return self.read_records(indices=sample_indices), sample_indices, corrected_weights

    def update_records(self, indices, update):
        pass
