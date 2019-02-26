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

from collections import OrderedDict

from rlgraph import get_backend
from rlgraph.components.memories.memory import Memory
from rlgraph.utils import util
from rlgraph.utils.execution_util import define_by_run_unflatten
from rlgraph.utils.util import get_batch_size
from rlgraph.utils.decorators import rlgraph_api

if get_backend() == "tf":
    import tensorflow as tf
elif get_backend() == "pytorch":
    import numpy as np
    import torch


class RingBuffer(Memory):
    """
    Simple ring-buffer to be used for on-policy sampling based on sample count
    or episodes. Fetches most recently added memories.

    API:
        get_episodes(num_episodes) -> Returns `num_episodes` episodes from the memory.
    """
    def __init__(self, capacity=1000, scope="ring-buffer", **kwargs):
        super(RingBuffer, self).__init__(capacity, scope=scope, **kwargs)

        self.index = None
        self.states = None
        self.num_episodes = None
        self.episode_indices = None
        self.flat_record_space = None

    def create_variables(self, input_spaces, action_space=None):
        super(RingBuffer, self).create_variables(input_spaces, action_space)

        # Record space must contain 'terminals' for a ring buffer memory.
        assert 'terminals' in self.record_space
        self.index = self.get_variable(name="index", dtype=int, trainable=False, initializer=0)
        # Num episodes present.
        self.num_episodes = self.get_variable(name="num-episodes", dtype=int, trainable=False, initializer=0)

        # Terminal indices contiguously arranged.
        self.episode_indices = self.get_variable(name="episode-indices", shape=(self.capacity,),
                                                 dtype=int, trainable=False)

    @rlgraph_api(flatten_ops=True)
    def _graph_fn_insert_records(self, records):
        if get_backend() == "tf":
            num_records = get_batch_size(records["terminals"])
            index = self.read_variable(self.index)

            # Episodes before inserting these records.
            prev_num_episodes = self.read_variable(self.num_episodes)
            update_indices = tf.range(start=index, limit=index + num_records) % self.capacity

            # Episodes previously existing in the range we inserted to as indicated
            # by count of terminals in the that slice.
            insert_terminal_slice = self.read_variable(self.record_registry['terminals'], update_indices)

            # Shift episode indices.
            with tf.control_dependencies([update_indices, index, prev_num_episodes, insert_terminal_slice]):
                index_updates = []

                # Newly inserted episodes.
                inserted_episodes = tf.reduce_sum(input_tensor=tf.cast(records['terminals'], dtype=tf.int32), axis=0)
                episodes_in_insert_range = tf.reduce_sum(
                    input_tensor=tf.cast(insert_terminal_slice, dtype=tf.int32), axis=0
                )
                num_episode_update = prev_num_episodes - episodes_in_insert_range + inserted_episodes

                # Shift contiguous episode indices.
                # prev_num_episodes = tf.Print(prev_num_episodes, [inserted_episodes, episodes_in_insert_range], summarize=100,
                #                              message="inserted_episodes, episodes in range = ")
                index_updates.append(self.assign_variable(
                        ref=self.episode_indices[:prev_num_episodes - episodes_in_insert_range],
                        value=self.episode_indices[episodes_in_insert_range:prev_num_episodes]
                ))

                # Insert new episodes starting at previous count minus the ones we removed,
                # ending at previous count minus removed + inserted.
                slice_start = prev_num_episodes - episodes_in_insert_range
                slice_end = num_episode_update

            # Update indices and size.
            with tf.control_dependencies(index_updates):
                index_updates = []

                # Actually update indices.
                mask = tf.boolean_mask(tensor=update_indices, mask=records['terminals'])
                # mask = tf.Print(mask, [self.episode_indices, update_indices, mask, slice_start, slice_end],
                #                 summarize=100, message="update, mask, start, end")
                index_updates.append(self.assign_variable(
                    ref=self.episode_indices[slice_start:slice_end],
                    value=mask
                ))

                # Assign final new episode count.
                index_updates.append(self.assign_variable(self.num_episodes, num_episode_update))

                index_updates.append(self.assign_variable(ref=self.index, value=(index + num_records) % self.capacity))
                update_size = tf.minimum(x=(self.read_variable(self.size) + num_records), y=self.capacity)
                index_updates.append(self.assign_variable(self.size, value=update_size))

            # Updates all the necessary sub-variables in the record.
            with tf.control_dependencies(index_updates):
                record_updates = []
                for key in self.record_registry:
                    record_updates.append(self.scatter_update_variable(
                        variable=self.record_registry[key],
                        indices=update_indices,
                        updates=records[key]
                    ))

            # Nothing to return.
            with tf.control_dependencies(control_inputs=record_updates):
                return tf.no_op()
        elif get_backend() == "pytorch":
            # TODO: Unclear if we should do this in numpy and then convert to torch once we sample.
            num_records = get_batch_size(records["terminals"])
            update_indices = torch.arange(self.index, self.index + num_records) % self.capacity

            # Newly inserted episodes.
            inserted_episodes = torch.sum(records['terminals'].int(), 0)

            # Episodes previously existing in the range we inserted to as indicated
            # by count of terminals in the that slice.
            episodes_in_insert_range = 0
            # Count terminals in inserted range.
            for index in update_indices:
                episodes_in_insert_range += int(self.record_registry["terminals"][index])
            num_episode_update = self.num_episodes - episodes_in_insert_range + inserted_episodes
            self.episode_indices[:self.num_episodes - episodes_in_insert_range] = \
                self.episode_indices[episodes_in_insert_range:self.num_episodes]

            # Insert new episodes starting at previous count minus the ones we removed,
            # ending at previous count minus removed + inserted.
            slice_start = self.num_episodes - episodes_in_insert_range
            slice_end = num_episode_update

            byte_terminals = records["terminals"].byte()
            mask = torch.masked_select(update_indices, byte_terminals)
            self.episode_indices[slice_start:slice_end] = mask

            # Update indices.
            self.num_episodes = int(num_episode_update)
            self.index = (self.index + num_records) % self.capacity
            self.size = min(self.size + num_records, self.capacity)

            # Updates all the necessary sub-variables in the record.
            for key in self.record_registry:
                for i, val in zip(update_indices, records[key]):
                    self.record_registry[key][i] = val

            # The TF version returns no-op, return None so return-val inference system does not throw error.
            return None

    @rlgraph_api
    def _graph_fn_get_records(self, num_records=1):
        if get_backend() == "tf":
            stored_records = self.read_variable(self.size)
            available_records = tf.minimum(x=num_records, y=stored_records)
            index = self.read_variable(self.index)
            indices = tf.range(start=index - available_records, limit=index) % self.capacity
            return self._read_records(indices=indices)
        elif get_backend() == "pytorch":
            available_records = min(num_records, self.size)
            indices = np.arange(self.index - available_records, self.index) % self.capacity
            records = OrderedDict()

            for name, variable in self.record_registry.items():
                records[name] = self.read_variable(variable, indices, dtype=
                                                   util.convert_dtype(self.flat_record_space[name].dtype, to="pytorch"),
                                                   shape=self.flat_record_space[name].shape)

            records = define_by_run_unflatten(records)
            return records

    @rlgraph_api(ok_to_overwrite=True)
    def _graph_fn_get_episodes(self, num_episodes=1):
        if get_backend() == "tf":
            stored_episodes = self.read_variable(self.num_episodes)
            available_episodes = tf.minimum(x=num_episodes, y=stored_episodes)

            # Say we have two episodes with this layout:
            # terminals = [0 0 1 0 1]
            # episode_indices = [2, 4]
            # If we want to fetch the most recent episode, the start index is:
            # stored_episodes - 1 - num_episodes = 2 - 1 - 1 = 0, which points to buffer index 2
            # The next episode starts one element after this, hence + 1.
            # However, this points to index -1 if stored_episodes = available_episodes,
            # in this case we want start = 0 to get everything.
            start = tf.cond(
                pred=tf.equal(x=stored_episodes, y=available_episodes),
                true_fn=lambda: 0,
                false_fn=lambda: self.episode_indices[stored_episodes - available_episodes - 1] + 1
            )
            # End index is just the pointer to the most recent episode.
            limit = self.episode_indices[stored_episodes - 1]

            limit += tf.where(condition=(start < limit), x=0, y=self.capacity - 1)
            # limit = tf.Print(limit, [stored_episodes, start, limit], summarize=100, message="start | limit")
            indices = tf.range(start=start, limit=limit + 1) % self.capacity
            return self._read_records(indices=indices)
        elif get_backend() == "pytorch":
            stored_episodes = self.num_episodes
            available_episodes = min(num_episodes, self.num_episodes)

            if stored_episodes == available_episodes:
                start = 0
            else:
                start = self.episode_indices[stored_episodes - available_episodes - 1] + 1

            # End index is just the pointer to the most recent episode.
            limit = self.episode_indices[stored_episodes - 1]
            if start >= limit:
                limit += self.capacity - 1
            indices = torch.arange(start, limit + 1) % self.capacity

            records = OrderedDict()
            for name, variable in self.record_registry.items():
                records[name] = self.read_variable(variable, indices, dtype=
                                                   util.convert_dtype(self.flat_record_space[name].dtype, to="pytorch"),
                                                   shape=self.flat_record_space[name].shape)
            records = define_by_run_unflatten(records)
            return records
