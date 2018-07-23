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

from yarl import get_backend
from yarl.components.memories.memory import Memory
from yarl.spaces.space_utils import sanity_check_space

if get_backend() == "tf":
    import tensorflow as tf


class FIFOQueue(Memory):
    """
    A wrapper for a simple in-graph FIFOQueue.
    """
    def __init__(self, **kwargs):
        super(FIFOQueue, self).__init__(scope=kwargs.pop("scope", "fifo-queue"), **kwargs)

        # Holds the actual backend-specific queue object.
        self.queue = None

    def create_variables(self, input_spaces, action_space):
        # Overwrite parent version as we don't need a custom registry.
        in_space = input_spaces["insert_records"][0]

        # Make sure all input-records have a batch rank and determine the shapes and dtypes.
        shapes = list()
        dtypes = list()
        names = list()
        for k, v in in_space.flatten().items():
            sanity_check_space(v, must_have_batch_rank=True)
            shapes.append(v.get_shape(with_batch_rank=True))
            dtypes.append(v.dtype)
            names.append(k)

        # Construct the wrapped FIFOQueue object.
        if get_backend() == "tf":
            self.queue = tf.FIFOQueue(
                capacity=self.capacity,
                dtypes=dtypes,
                shapes=shapes,
                names=names
            )

    def _graph_fn_insert_records(self, records):
        return self.queue.enqueue_many(records)

    def _graph_fn_get_records(self, num_records):
        return self.queue.dequeue_many(num_records)
