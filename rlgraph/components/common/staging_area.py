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
from rlgraph.components.component import Component
from rlgraph.utils.ops import flatten_op
from rlgraph.utils.util import dtype as dtype_

if get_backend() == "tf":
    import tensorflow as tf


class StagingArea(Component):
    """
    Stages an arbitrary number of incoming ops for next-step processing.
    This allows for more efficient handling of dequeued (or otherwise pipelined) data: The data can
    be prepared and then staged while a training step is still taking place, the next training step can then
    immediately take the staged data, aso.asf..
    """
    def __init__(self, scope="staging-area", **kwargs):
        super(StagingArea, self).__init__(scope=scope, **kwargs)

        # The actual backend-dependent StagingArea object.
        self.area = None
        # List of lists of flat keys of all input DataOps.
        self.flat_keys = list()

        self.define_api_method(name="stage", func=self._graph_fn_stage)
        #self.define_api_method(name="unstage", func=self._graph_fn_unstage)

    def create_variables(self, input_spaces, action_space=None):
        # Store the original structure for later recovery.
        dtypes = list()
        shapes = list()
        idx = 0
        while True:
            key = "inputs[{}]".format(idx)
            if key not in input_spaces:
                break
            dtypes.append(dtype_(input_spaces[key].dtype))
            shapes.append(input_spaces[key].get_shape(with_batch_rank=True, with_time_rank=True))
            idx += 1

        if get_backend() == "tf":
            self.area = tf.contrib.staging.StagingArea(dtypes, shapes)

    def _graph_fn_stage(self, *inputs):
        """
        Stages all incoming ops (after flattening them).

        Args:
            inputs (DataOp): The incoming ops to be (flattened and) staged.

        Returns:
            DataOp: The staging op.
        """
        # Flatten inputs and stage them.
        # TODO: Build equivalent to nest.flatten ()
        flattened_ops = list()
        for input_ in inputs:
            flat_list = list(flatten_op(input_).values())
            flattened_ops.extend(flat_list)
        return self.area.put(flattened_ops)

    """
    def _graph_fn_unstage(self):
        ""
        Unstages (and unflattens) all staged data.

        Returns:
            Tuple[DataOp]: All previously staged ops.
        ""
        unstaged_data = self.area.get()
        unflattened_data = list()
        # Unflatten all data and return.
        for flat_key_list, item in zip(self.flat_keys, unstaged_data):
            flat_data_op = FlattenedDataOp()
            unflattened_data.append(unflatten_op())

        return tuple(unflattened_data)
    """