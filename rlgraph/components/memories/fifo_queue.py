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
from rlgraph.utils.util import dtype as dtype_
from rlgraph.utils.decorators import rlgraph_api

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

        # If record space given, overwrite the insert method as "must_be_complete=False".
        if self.record_space is not None:
            @rlgraph_api(component=self, must_be_complete=False, ok_to_overwrite=True)
            def _graph_fn_insert_records(self, records):
                flattened_records = flatten_op(records)
                flattened_stopped_records = {key: tf.stop_gradient(op) for key, op in flattened_records.items()}
                # Records is just one record.
                if self.only_insert_single_records is True:
                    return self.queue.enqueue(flattened_stopped_records)
                # Insert many records (with batch rank).
                else:
                    return self.queue.enqueue_many(flattened_stopped_records)

    def create_variables(self, input_spaces, action_space=None):
        # Overwrite parent's method as we don't need a custom registry.
        if self.record_space is None:
            self.record_space = input_spaces["records"]

        # Make sure all input-records have a batch rank and determine the shapes and dtypes.
        shapes = []
        dtypes = []
        names = []
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
            if self.reuse_variable_scope:
                shared_name = self.reuse_variable_scope + ("/" + self.scope if self.scope else "")
            else:
                shared_name = self.global_scope

            self.queue = tf.FIFOQueue(
                capacity=self.capacity,
                dtypes=dtypes,
                shapes=shapes,
                names=names,
                shared_name=shared_name
            )

    @rlgraph_api
    def _graph_fn_get_records(self, num_records=1):
        # Get the records as dict.
        record_dict = self.queue.dequeue_many(num_records)
        # Return a FlattenedDataOp.
        flattened_records = FlattenedDataOp(record_dict)
        # Add batch and (possible) time rank to output ops for the auto-Space-inference.
        flat_record_space = self.record_space.flatten()
        for flat_key, op in record_dict.items():
            if flat_record_space[flat_key].has_time_rank:
                op._batch_rank = 0
                op._time_rank = 1
                flattened_records[flat_key] = op
            else:
                op._batch_rank = 0
                flattened_records[flat_key] = op
        return flattened_records

    @rlgraph_api
    def _graph_fn_get_size(self):
        """
        Returns the current size of the queue.

        Returns:
            DataOp: The current size of the queue (how many items are in it).
        """
        return self.queue.size()
