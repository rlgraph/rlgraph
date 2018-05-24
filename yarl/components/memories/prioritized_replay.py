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

import tensorflow as tf
import numpy as np
from yarl.components.memories.memory import Memory
from yarl.components.memories.segment_tree import SegmentTree


class PrioritizedReplay(Memory):
    """
    Implements pure TensorFlow prioritized replay.
    """
    def __init__(
        self,
        capacity=1000,
        name="",
        scope="prioritized-replay",
        next_states=True,
        alpha=1.0
    ):
        super(PrioritizedReplay, self).__init__(capacity, name, scope)

        # Variables.
        self.index = None
        self.size = None
        self.states = None
        self.next_states = next_states
        assert alpha > 0.0
        # Priority weight.
        self.alpha = alpha
        self.max_priority = 1.0

        self.define_inputs("update")
        self.define_outputs("update_records")

        self.add_computation(
            inputs="update",
            outputs="update_records",
            method=self._computation_update_records,
            flatten_ops=False
        )

    def create_variables(self, input_spaces):
        super(PrioritizedReplay, self).create_variables(input_spaces)

        # Record space must contain 'terminal' for a replay memory.
        assert 'terminal' in self.record_space

        # Main buffer index.
        self.index = self.get_variable(name="index", dtype=int, trainable=False, initializer=0)
        # Number of elements present.
        self.size = self.get_variable(name="size", dtype=int, trainable=False, initializer=0)

        # Segment tree must be full binary tree.
        self.priority_capacity = 1
        while self.priority_capacity < self.capacity:
            self.priority_capacity *= 2

        # 1. Create a variable for a sum-segment tree.
        self.sum_segment_buffer = self.get_variable(
                name="sum-segment-tree",
                shape=(2 * self.priority_capacity,),
                dtype=tf.float32,
                trainable=False
        )
        self.sum_segment_tree = SegmentTree(self.sum_segment_buffer, self.priority_capacity)

        # 2. Create a variable for a min-segment tree.
        self.min_segment_buffer = self.get_variable(
                name="min-segment-tree",
                dtype=tf.float32,
                trainable=False,
                initializer=tf.constant_initializer(np.full((2 * self.priority_capacity,), float('inf')))
        )
        self.min_segment_tree = SegmentTree(self.min_segment_buffer, self.priority_capacity)

        if self.next_states:
            assert 'states' in self.record_space

            # Next states are not represented as explicit keys in the registry
            # as this would cause extra memory overhead.
            self.states = []
            for state in self.record_space["states"].keys():
                self.states.append('/states/{}'.format(state))

    def _computation_insert(self, records):
        # TODO insert is same as replay, although we may want to check for priority here ->
        # depends on usage

        # TODO override priority values with max value
        num_records = tf.shape(input=records['/terminal'])[0]
        index = self.read_variable(self.index)
        update_indices = tf.range(start=index, limit=index + num_records) % self.capacity

        # Updates all the necessary sub-variables in the record.
        # update_indices = tf.Print(update_indices, [update_indices, index, num_records], summarize=100,
        #                           message='Update indices / index / num records = ')
        record_updates = list()
        for key in self.record_registry:
            record_updates.append(self.scatter_update_variable(
                variable=self.record_registry[key],
                indices=update_indices,
                updates=records[key]
            ))

        # Update indices and size.
        with tf.control_dependencies(control_inputs=record_updates):
            index_updates = list()
            index_updates.append(self.assign_variable(variable=self.index, value=(index + num_records) % self.capacity))
            update_size = tf.minimum(x=(self.read_variable(self.size) + num_records), y=self.capacity)
            index_updates.append(self.assign_variable(self.size, value=update_size))

        # Nothing to return.
        with tf.control_dependencies(control_inputs=index_updates):
            return tf.no_op()

    def _computation_get_records(self, num_records):
        # 1. execute sampling loop via
        # - sampling probability mass
        # - obtaining index from prefix sum
        # -> can be vectorized?
        pass

    def _computation_update_records(self, update):
        pass
