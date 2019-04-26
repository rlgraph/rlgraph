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

from rlgraph.utils import SMALL_NUMBER
from rlgraph.utils.specifiable import Specifiable
from rlgraph.components.helpers.mem_segment_tree import MemSegmentTree, MinSumSegmentTree
from rlgraph.execution.ray.ray_util import ray_decompress


class ApexMemory(Specifiable):
    """
    Apex prioritized replay implementing compression.
    """
    def __init__(self, state_space=None, action_space=None, capacity=1000, alpha=1.0, beta=1.0):
        """
        Args:
            state_space (dict): State spec.
            action_space (dict): Actions spec.
            capacity (int): Max capacity.
            alpha (float): Initial weight.
            beta (float): Prioritisation factor.
        """
        super(ApexMemory, self).__init__()

        self.state_space = state_space
        self.action_space = action_space
        self.container_actions = isinstance(action_space, dict)
        self.memory_values = []
        self.index = 0
        self.capacity = capacity
        self.size = 0
        self.max_priority = 1.0
        self.alpha = alpha
        self.beta = beta

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
        if record[5] is not None:
            self.merged_segment_tree.insert(self.index, record[5] ** self.alpha)
        else:
            self.merged_segment_tree.insert(self.index, self.max_priority ** self.alpha)

        # Update indices.
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def read_records(self, indices):
        """
        Obtains record values for the provided indices.

        Args:
            indices (ndarray): Indices to read. Assumed to be not contiguous.

        Returns:
             dict: Record value dict.
        """
        states = []
        if self.container_actions:
            actions = {k: [] for k in self.action_space.keys()}
        else:
            actions = []
        rewards = []
        terminals = []
        next_states = []
        for index in indices:
            state, action, reward, terminal, next_state, weight = self.memory_values[index]
            states.append(ray_decompress(state))

            if self.container_actions:
                for name in self.action_space.keys():
                    actions[name].append(action[name])
            else:
                actions.append(action)
            rewards.append(reward)
            terminals.append(terminal)
            next_states.append(ray_decompress(next_state))

        if self.container_actions:
            for name in self.action_space.keys():
                actions[name] = np.squeeze(np.array(actions[name]))
        else:
            actions = np.array(actions)
        return dict(
            states=np.asarray(states),
            actions=actions,
            rewards=np.asarray(rewards),
            terminals=np.asarray(terminals),
            next_states=np.asarray(next_states)
        )

    def get_records(self, num_records):
        indices = []
        prob_sum = self.merged_segment_tree.sum_segment_tree.get_sum(0, self.size)
        samples = np.random.random(size=(num_records,)) * prob_sum
        for sample in samples:
            indices.append(self.merged_segment_tree.sum_segment_tree.index_of_prefixsum(prefix_sum=sample))

        sum_prob = self.merged_segment_tree.sum_segment_tree.get_sum()
        min_prob = self.merged_segment_tree.min_segment_tree.get_min_value() / sum_prob + SMALL_NUMBER
        max_weight = (min_prob * self.size) ** (-self.beta)
        weights = []
        for index in indices:
            sample_prob = self.merged_segment_tree.sum_segment_tree.get(index) / sum_prob
            weight = (sample_prob * self.size) ** (-self.beta)
            weights.append(weight / max_weight)

        return self.read_records(indices=indices), np.asarray(indices), np.asarray(weights)

    def update_records(self, indices, update):
        for index, loss in zip(indices, update):
            self.merged_segment_tree.insert(index, loss ** self.alpha)
            self.max_priority = max(self.max_priority, loss)
