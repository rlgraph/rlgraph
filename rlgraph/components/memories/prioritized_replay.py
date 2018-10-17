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

import tensorflow as tf
import numpy as np

from rlgraph.components.memories.memory import Memory
from rlgraph.components.helpers.segment_tree import SegmentTree
from rlgraph.utils.decorators import rlgraph_api
from rlgraph.utils.ops import FlattenedDataOp
from rlgraph.utils.util import get_batch_size


class PrioritizedReplay(Memory):
    """
    Implements pure TensorFlow prioritized replay.

    API:
        update_records(indices, update) -> Updates the given indices with the given priority scores.
    """
    def __init__(self, capacity=1000, alpha=1.0, beta=0.0, scope="prioritized-replay", **kwargs):
        """
        Args:
            next_states (bool): Whether to include s' in the return values of the out-Socket "get_records".
            alpha (float): Degree to which prioritization is applied, 0.0 implies no
                prioritization (uniform), 1.0 full prioritization.
            beta (float): Importance weight factor, 0.0 for no importance correction, 1.0
                for full correction.
        """
        super(PrioritizedReplay, self).__init__(capacity, scope=scope, **kwargs)

        # Variables.
        self.index = None
        self.size = None
        self.max_priority = None
        self.sum_segment_buffer = None
        self.sum_segment_tree = None
        self.min_segment_buffer = None
        self.min_segment_tree = None

        # List of flattened keys in our state Space.
        self.flat_state_keys = None

        self.priority_capacity = 0

        # TODO check if we allow 0.0 as well.
        assert alpha > 0.0
        # Priority weight.
        self.alpha = alpha
        self.beta = beta

    def create_variables(self, input_spaces, action_space=None):
        super(PrioritizedReplay, self).create_variables(input_spaces, action_space)

        # Record space must contain 'terminals' for a replay memory.
        assert 'terminals' in self.record_space

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
                trainable=False,
                initializer=tf.zeros_initializer()
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

    @rlgraph_api(flatten_ops=True)
    def _graph_fn_insert_records(self, records):
        num_records = get_batch_size(records["terminals"])
        index = self.read_variable(self.index)
        update_indices = tf.range(start=index, limit=index + num_records) % self.capacity

        # Updates all the necessary sub-variables in the record.
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

        weight = tf.pow(x=self.max_priority, y=self.alpha)

        # Insert new priorities into segment tree.
        def insert_body(i):
            sum_insert = self.sum_segment_tree.insert(update_indices[i], weight, tf.add)
            with tf.control_dependencies(control_inputs=[sum_insert]):
                return i + 1

        def cond(i):
            return i < num_records

        with tf.control_dependencies(control_inputs=index_updates):
            sum_insert = tf.while_loop(cond=cond, body=insert_body, loop_vars=[0])

        def insert_body(i):
            min_insert = self.min_segment_tree.insert(update_indices[i], weight, tf.minimum)
            with tf.control_dependencies(control_inputs=[min_insert]):
                return i + 1

        def cond(i):
            return i < num_records

        with tf.control_dependencies(control_inputs=[sum_insert]):
            min_insert = tf.while_loop(cond=cond, body=insert_body, loop_vars=[0])

        # Nothing to return.
        with tf.control_dependencies(control_inputs=[min_insert]):
            return tf.no_op()

    @rlgraph_api
    def _graph_fn_get_records(self, num_records=1):
        # Sum total mass.
        current_size = self.read_variable(self.size)
        stored_elements_prob_sum = self.sum_segment_tree.reduce(start=0, limit=current_size - 1)

        # Sample the entire batch.
        sample = stored_elements_prob_sum * tf.random_uniform(shape=(num_records, ))

        # Sample by looking up prefix sum.
        sample_indices = tf.map_fn(fn=self.sum_segment_tree.index_of_prefixsum, elems=sample, dtype=tf.int32)
        # sample_indices = self.sum_segment_tree.index_of_prefixsum(sample)

        # Importance correction.
        total_prob = self.sum_segment_tree.reduce(start=0, limit=self.priority_capacity - 1)
        min_prob = self.min_segment_tree.get_min_value() / total_prob
        max_weight = tf.pow(x=min_prob * tf.cast(current_size, tf.float32), y=-self.beta)

        def importance_sampling_fn(sample_index):
            sample_prob = self.sum_segment_tree.get(sample_index) / stored_elements_prob_sum
            weight = tf.pow(x=sample_prob * tf.cast(current_size, tf.float32), y=-self.beta)

            return weight / max_weight

        corrected_weights = tf.map_fn(
            fn=importance_sampling_fn,
            elems=sample_indices,
            dtype=tf.float32
        )
        # sample_indices = tf.Print(sample_indices, [sample_indices, self.sum_segment_tree.values], summarize=1000,
        #                           message='sample indices, segment tree values = ')
        return self._read_records(indices=sample_indices), sample_indices, corrected_weights

    @rlgraph_api(must_be_complete=False)
    def _graph_fn_update_records(self, indices, update):
        num_records = get_batch_size(indices)
        max_priority = 0.0

        # Update has to be sequential.
        def insert_body(i, max_priority_):
            priority = tf.pow(x=update[i], y=self.alpha)

            sum_insert = self.sum_segment_tree.insert(
                index=indices[i],
                element=priority,
                insert_op=tf.add
            )
            min_insert = self.min_segment_tree.insert(
                index=indices[i],
                element=priority,
                insert_op=tf.minimum
            )
            # Keep track of current max priority element.
            max_priority_ = tf.maximum(x=max_priority_, y=priority)

            with tf.control_dependencies(control_inputs=[tf.group(sum_insert, min_insert)]):
                # TODO: This confuses the auto-return value detector.
                return i + 1, max_priority_

        def cond(i, max_priority_):
            return i < num_records - 1

        _, max_priority = tf.while_loop(
            cond=cond,
            body=insert_body,
            loop_vars=(0, max_priority)
        )

        assignment = self.assign_variable(ref=self.max_priority, value=max_priority)
        with tf.control_dependencies(control_inputs=[assignment]):
            return tf.no_op()

    def _read_records(self, indices):
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
        return records

