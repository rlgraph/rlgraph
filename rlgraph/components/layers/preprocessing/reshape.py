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

import numpy as np

from rlgraph.backend_system import get_backend
from rlgraph.components.layers.preprocessing import PreprocessLayer
from rlgraph.utils.ops import flatten_op, unflatten_op
from rlgraph.utils.util import get_rank, get_shape

if get_backend() == "tf":
    import tensorflow as tf


class ReShape(PreprocessLayer):
    """
    A simple reshape preprocessor that takes an input and reshapes it into a new shape. Use -1 in at most one
    rank to mark the flexible dimension.
    """
    def __init__(self, new_shapes, scope="reshape", **kwargs):
        """
        Args:
            new_shapes (Optional[Dict[str,Tuple[int]],Tuple[int]]): A dict of str/tuples or a single tuple
                specifying the new-shape(s) to use (for each auto key in case of a Container input Space).
                At most one of the ranks in any new_shape may be -1 to indicate flexibility in that dimension
                (e.g. the batch rank).
        """
        super(ReShape, self).__init__(scope=scope, add_auto_key_as_first_param=True, **kwargs)

        # The new shape specifications.
        self.new_shapes = new_shapes

        # The output spaces after preprocessing (per flat-key).
        self.output_spaces = None

    def get_preprocessed_space(self, space):
        ret = dict()
        for key, value in space.flatten().items():
            if isinstance(self.new_shapes, dict):
                #TODO: continue here!
                ret[key] = FloatBox(shape=self.new_shapes[key], add_batch_rank=value.has_batch_rank)
            else:
                ret[key] = self.new_shapes
        return unflatten_op(ret)

    def check_input_spaces(self, input_spaces, action_space=None):
        super(ReShape, self).check_input_spaces(input_spaces, action_space)

        # Check whether our input space has-batch or not and store this information here.
        in_space = input_spaces["inputs"]  # type: Dict

        # Store the mapped output Spaces (per flat key).
        self.output_spaces = flatten_op(self.get_preprocessed_space(in_space))

    def _graph_fn_apply(self, key, inputs):
        """
        Reshapes the input to the specified new shape.

        Args:
            inputs (SingleDataOp): The input to reshape.

        Returns:
            SingleDataOp: The reshaped input.
        """
        new_shape = self.new_shapes[key] if isinstance(self.new_shapes, dict) else self.new_shapes
        if self.backend == "python" or get_backend() == "python":
            reshaped = np.reshape(inputs, newshape=new_shape)
            return reshaped

        elif get_backend() == "tf":
            return tf.reshape(inputs, shape=new_shape, name="reshaped")

