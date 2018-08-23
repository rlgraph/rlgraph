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
from rlgraph.execution.ray.ray_util import ray_decompress


# TODO do not delete yet, will use this as basis for optimized env fragments
class LegacyApexMemory(Specifiable):
    """
    Apex prioritized replay implementing compression.
    """
    def __init__(self, capacity=1000, alpha=1.0, beta=1.0, n_step_adjustment=1):
        """
        TODO: documentation.
        Args:
            capacity ():
            alpha ():
            beta ():
            n_step_adjustment ():
        """
        super(LegacyApexMemory, self).__init__()

        self.memory_values = []
        self.index = 0
        self.capacity = capacity
        self.size = 0
        self.max_priority = 1.0
        self.alpha = alpha
        self.beta = beta
        assert n_step_adjustment > 0, "ERROR: n-step adjustment must be at least 1 where 1 corresponds" \
            "to the direct next state."
        self.n_step_adjustment = n_step_adjustment

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

        # Weights. # TODO this is problematic due to index not existing.
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
        states = list()
        actions = list()
        rewards = list()
        terminals = list()
        next_states = list()
        for index in indices:
            # TODO remove as array casts if they dont help.
            state, action, reward, terminal, weight = self.memory_values[index]
            decompressed_state = np.array(ray_decompress(state), copy=False)
            states.append(decompressed_state)
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            terminals.append(terminal)

            decompressed_next_state = decompressed_state
            # If terminal -> just use same state, already decompressed
            if terminal:
                next_states.append(decompressed_next_state)
            else:
                # Otherwise advance until correct next state or terminal.
                next_state = decompressed_next_state
                for i in range_(self.n_step_adjustment):
                    next_index = (index + i + 1) % self.size
                    next_state, _, _, terminal, _ = self.memory_values[next_index]
                    if terminal:
                        break
                next_states.append(np.array(ray_decompress(next_state), copy=False))

        return dict(
            states=np.array(states),
            actions=np.array(actions),
            rewards=np.array(rewards),
            terminals=np.array(terminals),
            next_states=np.array(next_states)
        )

    def get_records(self, num_records):
        indices = []
        # Ensure we always have n-next states.
        # TODO potentially block this if size - 1 - nstep < 1?
        prob_sum = self.merged_segment_tree.sum_segment_tree.get_sum(0, self.size - 1 - self.n_step_adjustment)
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

        indices = np.array(indices, copy=False)
        return self.read_records(indices=indices), indices, np.array(weights, copy=False)

    def update_records(self, indices, update):
        for index, loss in zip(indices, update):
            priority = np.power(loss, self.alpha)
            self.merged_segment_tree.insert(index, priority)
            self.max_priority = max(self.max_priority, priority)

