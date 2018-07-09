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
from yarl.components.memories.segment_tree import SegmentTree
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

        self.sum_segment_tree = MemSegmentTree(None)
        self.min_segment_tree = MemSegmentTree(None)

        self.default_new_weight = np.power(self.max_priority, self.alpha)

    def create_variables(self, input_spaces, action_space):
        # Store our record-space for convenience.
        self.record_space = input_spaces["insert_records"][0]

        # Create the main memory as a flattened OrderedDict from any arbitrarily nested Space.
        self.record_registry = get_list_registry(self.record_space)

    def insert_records(self, records):
        num_records = len(records["/terminals"])
        update_indices = np.arange(start=self.index, stop=self.index + num_records) % self.capacity

        # Update record registry.
        # record_updates = list()
        # for key in self.record_registry:
        #     record_updates.append(self.scatter_update_variable(
        #         variable=self.record_registry[key],
        #         indices=update_indices,
        #         updates=records[key]
        #     ))
        #

        # Update indices
        self.index = (self.index + num_records) % self.capacity
        self.size = min(self.size + num_records, self.capacity)

        # Insert into segment trees.
        for i in xrange(num_records):
            self.sum_segment_tree.insert(update_indices[i], self.default_new_weight, operator.add)
            self.min_segment_tree.insert(update_indices[i], self.default_new_weight, min)

    def get_records(self, num_records):
        pass

    def update_records(self, indices, update):
        pass
