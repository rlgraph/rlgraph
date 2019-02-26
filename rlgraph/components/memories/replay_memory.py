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
    import torch
    import numpy as np


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
        self.states = None
        self.flat_record_space = None

    def create_variables(self, input_spaces, action_space=None):
        super(ReplayMemory, self).create_variables(input_spaces, action_space)

        # Record space must contain 'terminals' for a replay memory.
        assert 'terminals' in self.record_space
        # Main buffer index.
        self.index = self.get_variable(name="index", dtype=int, trainable=False, initializer=0)

    @rlgraph_api(flatten_ops=True)
    def _graph_fn_insert_records(self, records):
        num_records = get_batch_size(records["terminals"])

        if get_backend() == "tf":
            # List of indices to update (insert from `index` forward and roll over at `self.capacity`).
            update_indices = tf.range(start=self.index, limit=self.index + num_records) % self.capacity

            # Updates all the necessary sub-variables in the record.
            record_updates = []
            for key in self.record_registry:
                record_updates.append(self.scatter_update_variable(
                    variable=self.record_registry[key],
                    indices=update_indices,
                    updates=records[key]
                ))

            # Update indices and size.
            with tf.control_dependencies(control_inputs=record_updates):
                index_updates = [self.assign_variable(ref=self.index, value=(self.index + num_records) % self.capacity)]
                update_size = tf.minimum(x=(self.read_variable(self.size) + num_records), y=self.capacity)
                index_updates.append(self.assign_variable(self.size, value=update_size))

            # Nothing to return.
            with tf.control_dependencies(control_inputs=index_updates):
                return tf.no_op()
        elif get_backend() == "pytorch":
            update_indices = torch.arange(self.index, self.index + num_records) % self.capacity
            for key in self.record_registry:
                for i, val in zip(update_indices, records[key]):
                    self.record_registry[key][i] = val
            self.index = (self.index + num_records) % self.capacity
            self.size = min(self.size + num_records, self.capacity)
            return None

    @rlgraph_api
    def _graph_fn_get_records(self, num_records=1):
        if get_backend() == "tf":
            size = self.read_variable(self.size)

            # Sample and retrieve a random range, including terminals.
            index = self.read_variable(self.index)
            indices = tf.random_uniform(shape=(num_records,), maxval=size, dtype=tf.int32)
            indices = (index - 1 - indices) % self.capacity

            # Return default importance weight one.
            return self._read_records(indices=indices), indices, tf.ones_like(tensor=indices, dtype=tf.float32)
        elif get_backend() == "pytorch":
            indices = []
            if self. size > 0:
                indices = np.random.choice(np.arange(0, self.size), size=int(num_records))
                indices = (self.index - 1 - indices) % self.capacity
            records = OrderedDict()
            for name, variable in self.record_registry.items():
                records[name] = self.read_variable(variable, indices, dtype=
                                                   util.convert_dtype(self.flat_record_space[name].dtype, to="pytorch"),
                                                   shape=self.flat_record_space[name].shape)
            records = define_by_run_unflatten(records)
            weights = torch.ones(indices.shape, dtype=torch.float32) if len(indices) > 0 \
                else torch.ones(1, dtype=torch.float32)
            return records, indices, weights
