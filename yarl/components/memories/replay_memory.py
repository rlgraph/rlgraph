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

from yarl.components.memories.memory import Memory
from yarl.spaces import Space
import numpy as np
import tensorflow as tf


class ReplayMemory(Memory):
    """
    Implements a standard replay memory to sample randomized batches.
    """

    def __init__(
        self,
        record_space,
        capacity=1000,
        name="",
        scope="replay-memory",
        sub_indexes=None,
        next_states=True,
        sequences=False
    ):
        super(ReplayMemory, self).__init__(record_space, capacity, name, scope, sub_indexes)
        self.next_states = next_states
        self.sequences = sequences

    def create_variables(self):
        # Main memory.
        self.replay_buffer = self.get_variable(
            name="replay_buffer",
            trainable=False,
            from_space=self.record_space
        )
        # Main buffer index.
        self.index = self.get_variable(
            name="index",
            dtype=int,
            trainable=False,
            from_space=None
        )

    def _computation_insert(self, records):
        num_records = tf.shape(records.keys()[0])
        update_indices = np.arange(start=self.index, stop=self.index + num_records) % self.capacity

        # Updates all the necessary sub-variables in the record.
        updates = list()
        # self.scatter_update_records(variables=self.replay_buffer)

        # Update main buffer index and episode index.
        tf.assign(ref=self.index, value=(self.index + num_records) % self.capacity)

        # Nothing to return
        return tf.no_op()

    def _computation_get_records(self, num_records):
        pass

    def _computation_get_sequences(self, num_sequences, sequence_length):
        pass
