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
from rlgraph.spaces import FloatBox
from rlgraph.utils.ops import flatten_op, unflatten_op
from rlgraph.utils.util import get_rank, get_shape

if get_backend() == "tf":
    import tensorflow as tf


class ReShape(PreprocessLayer):
    """
    A simple reshape preprocessor that takes an input and reshapes it into a new shape. Use the value -1 in (at most)
    one of the new-shape's rank to mark a flexible dimension.
    """
    def __init__(self, new_shape, scope="reshape", **kwargs):
        """
        Args:
            new_shape (Optional[Dict[str,Tuple[int]],Tuple[int]]): A dict of str/tuples or a single tuple
                specifying the new-shape(s) to use (for each auto key in case of a Container input Space).
                At most one of the ranks in any new_shape may be -1 to indicate flexibility in that dimension
                (e.g. the batch rank).
        """
        super(ReShape, self).__init__(scope=scope, add_auto_key_as_first_param=True, **kwargs)

        # The new shape specifications.
        self.new_shape = new_shape

        # The output spaces after preprocessing (per flat-key).
        self.output_spaces = None

    def get_preprocessed_space(self, space):
        ret = dict()
        for key, value in space.flatten().items():
            if isinstance(self.new_shape, dict):
                ret[key] = FloatBox(shape=self.new_shape[key], add_batch_rank=value.has_batch_rank)
            else:
                ret[key] = self.new_shape
        return unflatten_op(ret)

    def check_input_spaces(self, input_spaces, action_space=None):
        super(ReShape, self).check_input_spaces(input_spaces, action_space)

        # Check whether our input space has-batch or not and store this information here.
        in_space = input_spaces["preprocessing_inputs"]  # type: Dict

        # Store the mapped output Spaces (per flat key).
        self.output_spaces = flatten_op(self.get_preprocessed_space(in_space))

    def _graph_fn_apply(self, key, preprocessing_inputs):
        """
        Reshapes the input to the specified new shape.

        Args:
            preprocessing_inputs (SingleDataOp): The input to reshape.

        Returns:
            SingleDataOp: The reshaped input.
        """
        new_shape = self.new_shape[key] if isinstance(self.new_shape, dict) else self.new_shape
        if self.backend == "python" or get_backend() == "python":
            return np.reshape(preprocessing_inputs, newshape=new_shape)

        elif get_backend() == "tf":
            return tf.reshape(preprocessing_inputs, shape=new_shape, name="reshaped")
