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

from rlgraph import get_backend
from rlgraph.components.memories.memory import Memory
from rlgraph.spaces.space_utils import sanity_check_space
from rlgraph.utils.ops import FlattenedDataOp, flatten_op
from rlgraph.utils.util import dtype as dtype_, get_shape

if get_backend() == "tf":
    import tensorflow as tf


class FIFOQueue(Memory):
    """
    A wrapper for a simple in-graph FIFOQueue.
    """
    def __init__(self, record_space=None, only_insert_single_records=False, **kwargs):
        """
        Args:
            record_space (Space): The Space of a single record to be pushed to or pulled from the queue.
            only_insert_single_records (bool): Whether insertion will always only happen with single records.
                If True, will add a batch=1 rank to each to-be-inserted sample.
        """
        super(FIFOQueue, self).__init__(scope=kwargs.pop("scope", "fifo-queue"), **kwargs)

        # The record Space must be provided for clients of the Queue that only use it for retrieving records, but never
        # inserting any. This way, RLgraph cannot infer the input space itself.
        self.record_space = record_space

        self.only_insert_single_records = only_insert_single_records
        # Holds the actual backend-specific queue object.
        self.queue = None

        self.define_api_method("get_size", self._graph_fn_get_size)
        # If record space given, overwrite the insert method as "must_be_complete=False".
        if self.record_space is not None:
            self.define_api_method(
                "insert_records", self._graph_fn_insert_records, must_be_complete=False, ok_to_overwrite=True
            )
            #self.define_api_method(
            #    "insert_dummy_records", self._graph_fn_insert_dummy_records, must_be_complete=False
            #)

    def create_variables(self, input_spaces, action_space=None):
        # Overwrite parent's method as we don't need a custom registry.
        if self.record_space is None:
            self.record_space = input_spaces["records"]

        # Make sure all input-records have a batch rank and determine the shapes and dtypes.
        shapes = list()
        dtypes = list()
        names = list()
        for key, value in self.record_space.flatten().items():
            # TODO: what if single items come in without a time-rank? Then this check here will fail.
            # We are expecting single items. The incoming batch-rank is actually a time-rank: Add the batch rank.
            sanity_check_space(value, must_have_batch_rank=self.only_insert_single_records is False)
            shape = value.get_shape(with_time_rank=value.has_time_rank)
            shapes.append(shape)
            dtypes.append(dtype_(value.dtype))
            names.append(key)

        # Construct the wrapped FIFOQueue object.
        if get_backend() == "tf":
            self.queue = tf.FIFOQueue(
                capacity=self.capacity,
                dtypes=dtypes,
                shapes=shapes,
                names=names
            )

    def _graph_fn_insert_records(self, records):
        # records is just one record: Add batch rank.
        if self.only_insert_single_records is True:
            # TODO: simply try: self.queue.enqueue(flatten_op(records))
            return self.queue.enqueue(flatten_op(records))
            #records = FlattenedDataOp({
            #    flat_key: tf.expand_dims(tensor, axis=0) for flat_key, tensor in flatten_op(records).items()
            #})
        else:
            # Insert the records as FlattenedDataOp (dict).
            return self.queue.enqueue_many(flatten_op(records))

    #def _graph_fn_insert_dummy_records(self):
    #    records = self.record_space.sample(size=(10, 20))
    #    return self.queue.enqueue_many(flatten_op(records))

    def _graph_fn_get_records(self, num_records=1):
        # Get the records as dict.
        record_dict = self.queue.dequeue_many(num_records)
        # Return a FlattenedDataOp.
        flattened_records = FlattenedDataOp(record_dict)
        # Add batch and (possible) time rank to output ops for the auto-Space-inference.
        flat_record_space = self.record_space.flatten()
        for flat_key, op in record_dict.items():
            if flat_record_space[flat_key].has_time_rank:
                op = tf.placeholder_with_default(op, shape=(None, None) + get_shape(op)[2:])
                op._batch_rank = 0
                op._time_rank = 1
                flattened_records[flat_key] = op
            else:
                op._batch_rank = 0
                flattened_records[flat_key] = tf.placeholder_with_default(op, shape=(None,) + get_shape(op)[1:])
        return flattened_records

    def _graph_fn_get_size(self):
        """
        Returns the current size of the queue.

        Returns:
            DataOp: The current size of the queue (how many items are in it).
        """
        return self.queue.size()
