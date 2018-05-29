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
import re

from yarl.components.memories.memory import Memory
from yarl.utils.ops import DataOpDict, FlattenedDataOp


class ReplayMemory(Memory):
    """
    Implements a standard replay memory to sample randomized batches.
    """

    def __init__(
        self,
        capacity=1000,
        name="",
        scope="replay-memory",
        next_states=True
    ):
        super(ReplayMemory, self).__init__(capacity, name, scope)
        self.add_graph_fn(
            inputs="num_records",
            outputs="sample",
            method=self._graph_fn_get_records,
            flatten_ops=False
        )

        # Variables.
        self.index = None
        self.size = None
        self.states = None
        self.next_states = next_states

    def create_variables(self, input_spaces):
        super(ReplayMemory, self).create_variables(input_spaces)

        # Record space must contain 'terminal' for a replay memory.
        assert 'terminal' in self.record_space

        # Main buffer index.
        self.index = self.get_variable(name="index", dtype=int, trainable=False, initializer=0)
        # Number of elements present.
        self.size = self.get_variable(name="size", dtype=int, trainable=False, initializer=0)
        if self.next_states:
            assert 'states' in self.record_space

            # Next states are not represented as explicit keys in the registry
            # as this would cause extra memory overhead.
            self.states = ["/states{}".format(flat_key) for flat_key in self.record_space["states"].flatten().keys()]

    def _graph_fn_insert(self, records):
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

        # Nothing to return.
        with tf.control_dependencies(control_inputs=index_updates):
            return tf.no_op()

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

    def _graph_fn_get_records(self, num_records):
        indices = tf.range(start=0, limit=self.read_variable(self.size))

        # TODO When would we use a replay memory without next-states?
        if self.next_states:
            # Valid indices are non-terminal indices
            terminal_indices = self.read_variable(self.record_registry['/terminal'], indices=indices)
            terminal_indices = tf.Print(terminal_indices, [terminal_indices], summarize=100, message='terminal_indices = ')
            indices = tf.Print(indices, [indices], summarize=100, message='indices = ')
            mask = tf.logical_not(x=tf.cast(terminal_indices, dtype=tf.bool))
            # mask = tf.Print(mask, [mask], summarize=100, message= 'mask = ')
            indices = tf.boolean_mask(
                tensor=indices,
                mask=mask
            )

        # Choose with uniform probability from all valid indices.
        # TODO problem - if there are no valid indices, we cannot return anything
        num_valid_indices = tf.shape(input=indices)[0]
        probabilities = tf.ones(shape=[num_valid_indices, num_valid_indices])
        samples = tf.multinomial(logits=probabilities, num_samples=num_valid_indices)
        # samples = tf.Print(samples, [samples, tf.shape(samples)], summarize=100, message='samples / shape = ')

        # Gather sampled indices from all indices.
        sampled_indices = tf.gather(params=indices, indices=tf.cast(x=samples[0], dtype=tf.int32))
        # sampled_indices = tf.Print(sampled_indices, [sampled_indices], summarize=100, message='sampled indices = ')

        return self.read_records(indices=sampled_indices)
