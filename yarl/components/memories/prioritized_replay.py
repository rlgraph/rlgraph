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

import re
import tensorflow as tf
import numpy as np

from yarl.components.memories.memory import Memory
from yarl.components.memories.segment_tree import SegmentTree
from yarl.utils.ops import FlattenedDataOp


class PrioritizedReplay(Memory):
    # XXX: this could maybe later inherit from replay buffer, is kept separate for now
    # until design fully clear.

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

        self.define_inputs("indices", "update")
        self.define_outputs("sample_indices", "update_records")
        self.add_graph_fn(
            inputs="num_records",
            outputs=["sample", "sample_indices"],
            method=self._graph_fn_get_records,
            flatten_ops=False
        )

        self.add_graph_fn(
            inputs=["indices", "update"],
            outputs="update_records",
            method=self._graph_fn_update_records,
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

        self.max_priority = self.get_variable(name="max-priority", dtype=float, trainable=False, initializer=1.0)

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
                # Neutral element of min()
                shape=(2 * self.priority_capacity,),
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

    def _graph_fn_insert(self, records):
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
            index_updates.append(self.assign_variable(ref=self.index, value=(index + num_records) % self.capacity))
            update_size = tf.minimum(x=(self.read_variable(self.size) + num_records), y=self.capacity)
            index_updates.append(self.assign_variable(self.size, value=update_size))

        # Note: Cannot concurrently modify, so need iterative insert
        def insert_body(i):
            sum_insert = self.sum_segment_tree.insert(
                index=update_indices[i],
                element=self.max_priority,
                insert_op=tf.add
            )

            min_insert = self.min_segment_tree.insert(
                index=update_indices[i],
                element=self.max_priority,
                insert_op=tf.minimum
            )
            with tf.control_dependencies(control_inputs=[tf.group(sum_insert, min_insert)]):
                return i + 1

        def cond(i):
            return i < num_records - 1

        with tf.control_dependencies(control_inputs=index_updates):
            index = tf.while_loop(cond=cond, body=insert_body, loop_vars=[0])

        # Nothing to return.
        with tf.control_dependencies(control_inputs=[index]):
            return tf.no_op()

    def _graph_fn_get_records(self, num_records):
        # Sum total mass.
        current_size = self.read_variable(self.size)
        prob_sum = self.sum_segment_tree.reduce(start=0, limit=current_size - 1)

        # Sample the entire batch.
        sample = prob_sum * tf.random_uniform(shape=(num_records, ))

        # Vectorized search loop searches through tree in parallel.
        sample_indices = tf.map_fn(fn=self.sum_segment_tree.index_of_prefixsum, elems=sample)
        # sample_indices = self.sum_segment_tree.index_of_prefixsum(sample)
        print('sample indices = {}'.format(sample_indices))
        # TODO what about filtering terminals?
        # - Searching prefix sum/resampling too expensive.
        return self.read_records(indices=sample_indices)

    def read_records(self, indices):
        """
        Obtains record values for the provided indices.

        Args:
            indices (Union[ndarray,tf.Tensor]): Indices to read. Assumed to be not contiguous.

        Returns:
             FlattenedDataOp: Record value dict.
        """
        records = FlattenedDataOp()
        for name, variable in self.record_registry.items():
            records[name] = self.read_variable(variable, indices)
        if self.next_states:
            next_indices = (indices + 1) % self.capacity

            # Next states are read via index shift from state variables.
            for state_name in self.states:
                next_states = self.read_variable(self.record_registry[state_name], next_indices)
                next_state_name = re.sub(r'^/states/', "/next_states/", state_name)
                records[next_state_name] = next_states
        return records

    def _graph_fn_update_records(self, indices, update):
        num_records = tf.shape(input=indices)[0]
        max_priority = 0.0

        # Update has to be sequential.
        def insert_body(i, max_priority, assignments):
            with tf.control_dependencies(control_inputs=assignments):
                priority = tf.pow(x=update[i], y=self.alpha)
                assignments = self.sum_segment_tree.insert(
                    index=indices[i],
                    element=priority,
                    insert_op=tf.add
                )
                assignments.extend(self.min_segment_tree.insert(
                    index=indices[i],
                    element=priority,
                    insert_op=tf.minimum
                ))
                max_priority = tf.maximum(x=max_priority, y=priority)
            return i + 1, max_priority, assignments

        def cond(i, max_priority, assignments):
            return i < num_records - 1

        _, max_priority, assignments = tf.while_loop(cond=cond, body=insert_body, loop_vars=(0, max_priority, list()))

        assignments.append(self.assign_variable(ref=self.max_priority, value=max_priority))
        with tf.control_dependencies(control_inputs=assignments):
            return tf.no_op()

