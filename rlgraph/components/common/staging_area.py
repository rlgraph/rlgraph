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
from rlgraph.utils.decorators import rlgraph_api
from rlgraph.utils.ops import flatten_op, unflatten_op, FlattenedDataOp
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
    def __init__(self, num_data=1, device="/device:GPU:0", scope="staging-area", **kwargs):
        """
        Args:
            num_data (int): The number of data items to stage. Each item can be a ContainerDataOp (which
                will be flattened (stage) and unflattened (unstage) automatically).
        """
        super(StagingArea, self).__init__(
            graph_fn_num_outputs=dict(_graph_fn_unstage=num_data),
            device=device,
            scope=scope, **kwargs
        )

        # The actual backend-dependent StagingArea object.
        self.area = None
        # List of lists of flat keys of all input DataOps.
        self.flat_keys = list()

    def create_variables(self, input_spaces, action_space=None):
        # Store the original structure for later recovery.
        dtypes = list()
        shapes = list()
        idx = 0
        while True:
            key = "inputs[{}]".format(idx)
            if key not in input_spaces:
                break
            flat_keys = list()
            for flat_key, flat_space in input_spaces[key].flatten().items():
                dtypes.append(dtype_(flat_space.dtype))
                shapes.append(flat_space.get_shape(with_batch_rank=True, with_time_rank=True))
                flat_keys.append(flat_key)
            self.flat_keys.append(flat_keys)
            idx += 1

        if get_backend() == "tf":
            self.area = tf.contrib.staging.StagingArea(dtypes, shapes)

    @rlgraph_api
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
        stage_op = self.area.put(flattened_ops)
        return stage_op

    @rlgraph_api
    def _graph_fn_unstage(self):
        """
        Unstages (and unflattens) all staged data.

        Returns:
            Tuple[DataOp]: All previously staged ops.
        """
        unstaged_data = self.area.get()
        unflattened_data = list()
        idx = 0
        # Unflatten all data and return.
        for flat_key_list in self.flat_keys:
            flat_dict = FlattenedDataOp({flat_key: item for flat_key, item in zip(flat_key_list, unstaged_data[idx:idx + len(flat_key_list)])})
            unflattened_data.append(unflatten_op(flat_dict))
            idx += len(flat_key_list)

        return tuple(unflattened_data)
