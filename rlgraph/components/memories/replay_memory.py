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

from rlgraph.components.memories.memory import Memory
from rlgraph.utils.ops import FlattenedDataOp
from rlgraph.utils.util import get_batch_size
from rlgraph.utils.decorators import rlgraph_api


class ReplayMemory(Memory):
    """
    Implements a standard replay memory to sample randomized batches.
    """
    def __init__(
        self,
        capacity=1000,
        scope="replay-memory",
        **kwargs
    ):
        """
        Args:
            next_states (bool): If true include next states in the return values of the API-method "get_records".
        """
        super(ReplayMemory, self).__init__(capacity, scope=scope, **kwargs)

        self.index = None
        self.size = None
        self.states = None

    def create_variables(self, input_spaces, action_space=None):
        super(ReplayMemory, self).create_variables(input_spaces, action_space)

        # Record space must contain 'terminals' for a replay memory.
        assert 'terminals' in self.record_space

        # Main buffer index.
        self.index = self.get_variable(name="index", dtype=int, trainable=False, initializer=0)
        # Number of elements present.
        self.size = self.get_variable(name="size", dtype=int, trainable=False, initializer=0)

    @rlgraph_api(flatten_ops=True)
    def _graph_fn_insert_records(self, records):
        num_records = get_batch_size(records["terminals"])
        # List of indices to update (insert from `index` forward and roll over at `self.capacity`).
        update_indices = tf.range(start=self.index, limit=self.index + num_records) % self.capacity

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
            index_updates.append(self.assign_variable(ref=self.index, value=(self.index + num_records) % self.capacity))
            update_size = tf.minimum(x=(self.read_variable(self.size) + num_records), y=self.capacity)
            index_updates.append(self.assign_variable(self.size, value=update_size))

        # Nothing to return.
        with tf.control_dependencies(control_inputs=index_updates):
            return tf.no_op()

    @rlgraph_api
    def _graph_fn_get_records(self, num_records=1):
        size = self.read_variable(self.size)

        # Sample and retrieve a random range, including terminals.
        index = self.read_variable(self.index)
        indices = tf.random_uniform(shape=(num_records,), maxval=size, dtype=tf.int32)
        indices = (index - 1 - indices) % self.capacity

        # Return default importance weight one.
        return self._read_records(indices=indices), indices, tf.ones_like(tensor=indices, dtype=tf.float32)

