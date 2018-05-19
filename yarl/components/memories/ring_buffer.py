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
import tensorflow as tf


class RingBuffer(Memory):
    """
    Simple ring-buffer to be used for on-policy sampling based on sample count
    or episodes. Fetches most recently added memories.
    """
    def __init__(
        self,
        capacity=1000,
        name="",
        scope="ring-buffer",
        episode_semantics=False
    ):
        super(RingBuffer, self).__init__(capacity, name, scope)
        # Variables.
        self.index = None
        self.size = None
        self.states = None
        self.episode_semantics = episode_semantics
        if self.episode_semantics:
            self.num_episodes = None
            self.episode_indices = None

    def create_variables(self, input_spaces):
        super(RingBuffer, self).create_variables(input_spaces)

        # Main buffer index.
        self.index = self.get_variable(name="index", dtype=int, trainable=False, initializer=0)
        # Number of elements present.
        self.size = self.get_variable(name="size", dtype=int, trainable=False, initializer=0)

        if self.episode_semantics:
            # Num episodes present.
            self.num_episodes = self.get_variable(name="num-episodes", dtype=int, trainable=False, initializer=0)

            # Terminal indices contiguously arranged.
            self.episode_indices = self.get_variable(
                name="episode-indices",
                shape=(self.capacity,),
                dtype=int,
                trainable=False
            )

    def _computation_insert(self, records):
        num_records = tf.shape(input=records['/terminal'])[0]
        index = self.read_variable(self.index)
        update_indices = tf.range(start=index, limit=index + num_records) % self.capacity

        # Update indices and size.
        index_updates = list()
        if self.episode_semantics:
            # Episodes before inserting these records.
            prev_num_episodes = self.read_variable(self.num_episodes)

            # Newly inserted episodes.
            inserted_episodes = tf.reduce_sum(input_tensor=records['/terminal'], axis=0)

            # Episodes previously existing in the range we inserted to as indicated
            # by count of terminals in the that slice.
            insert_terminal_slice = self.read_variable(self.record_registry['/terminal'], update_indices)
            episodes_in_insert_range = tf.reduce_sum(input_tensor=insert_terminal_slice, axis=0)

            # prev_num_episodes = tf.Print(prev_num_episodes, [
            #     prev_num_episodes,
            #     episodes_in_insert_range,
            #     inserted_episodes],
            #     summarize=100, message='previous num eps / prev episodes in insert range / inserted eps = '
            # )
            num_episode_update = prev_num_episodes - episodes_in_insert_range + inserted_episodes

            # prev_num_episodes = tf.Print(prev_num_episodes, [prev_num_episodes, episodes_in_insert_range],
            #                             summarize=100, message='num eps, eps in insert range =')
            # Remove previous episodes in inserted range.
            index_updates.append(self.assign_variable(
                    variable=self.episode_indices[:prev_num_episodes + 1 - episodes_in_insert_range],
                    value=self.episode_indices[episodes_in_insert_range:prev_num_episodes + 1]
            ))

            # Insert new episodes starting at previous count minus the ones we removed,
            # ending at previous count minus removed + inserted.
            slice_start = prev_num_episodes - episodes_in_insert_range
            slice_end = num_episode_update
            update_indices = tf.Print(update_indices, [update_indices, tf.shape(update_indices)],
                                      summarize=100, message='\n update indices / shape = ')
            # slice_start = tf.Print(
            #     slice_start, [slice_start, slice_end, self.episode_indices],
            #     summarize=100,
            #     message='\n slice start/ slice end / episode indices before = '
            # )

            with tf.control_dependencies(index_updates):
                index_updates = list()
                mask = tf.boolean_mask(tensor=update_indices, mask=records['/terminal'])
                mask = tf.Print(mask, [mask, update_indices, records['/terminal']], summarize=100,
                    message='\n mask /  update indices / records-terminal')

                index_updates.append(self.assign_variable(
                    variable=self.episode_indices[slice_start:slice_end],
                    value=mask
                ))
                # num_episode_update = tf.Print(num_episode_update, [num_episode_update, self.episode_indices],
                #     summarize=100,  message='\n num episodes / episode indices after: ')

                # Assign final new episode count.
                index_updates.append(self.assign_variable(self.num_episodes, num_episode_update))

        index_updates.append(self.assign_variable(variable=self.index, value=(index + num_records) % self.capacity))
        update_size = tf.minimum(x=(self.read_variable(self.size) + num_records), y=self.capacity)
        index_updates.append(self.assign_variable(self.size, value=update_size))

        # Updates all the necessary sub-variables in the record.
        with tf.control_dependencies(index_updates):
            record_updates = list()
            for key in self.record_registry:
                record_updates.append(self.scatter_update_variable(
                    variable=self.record_registry[key],
                    indices=update_indices,
                    updates=records[key]
                ))

        # Nothing to return.
        with tf.control_dependencies(control_inputs=record_updates):
            return tf.no_op()

    def read_records(self, indices):
        """
        Obtains record values for the provided indices.

        Args:
            indices Union[ndarray, tf.Tensor]: Indices to read. Assumed to be not contiguous.

        Returns:
             dict: Record value dict.
        """
        records = dict()
        for name, variable in self.record_registry.items():
            records[name] = self.read_variable(variable, indices)
        return records

    def _computation_get_records(self, num_records):
        stored_records = self.read_variable(self.size)
        index = self.read_variable(self.index)

        # We do not return duplicate records here.
        available_records = tf.minimum(x=stored_records, y=num_records)
        indices = tf.range(start=index - 1 - available_records, limit=index - 1) % self.capacity

        # TODO zeroing out terminals per flag?
        return self.read_records(indices=indices)

    def _computation_get_episodes(self, num_episodes):
        stored_episodes = self.read_variable(self.num_episodes)
        available_episodes = tf.minimum(x=num_episodes, y=stored_episodes)

        start = self.episode_indices[stored_episodes - available_episodes - 1] + 1
        limit = self.episode_indices[stored_episodes - 1]
        limit += tf.where(condition=(start < limit), x=0, y=self.capacity)

        indices = tf.range(start=start, limit=limit) % self.capacity

        return self.read_records(indices=indices)

    def get_variables(self, names=None):
        if self.episode_semantics:
            return [self.size, self.index, self.num_episodes, self.episode_indices]
        else:
            return [self.size, self.index]