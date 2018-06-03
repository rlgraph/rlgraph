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

from yarl import backend
from yarl.utils.util import force_list

from .nn_layer import NNLayer


class ConcatLayer(NNLayer):
    """
    A simple concatenation layer wrapper. The ConcatLayer is a Layer without sub-components but with n
    inputs and 1 output, where the in-Sockets's data are concatenated into one out-Socket by our GraphFunction.
    """
    def __init__(self, axis=-1, scope="concat-layer", num_graph_fn_inputs=2, **kwargs):
        """
        Args:
            axis (int): The axis along which to concatenate. Use negative numbers to count from end.
                All inputs to this layer must have the same shape, except for the `axis` rank.
            num_graph_fn_inputs (int): The number of inputs to concatenate (this is how many in-Sockets will be created).
        """
        # Set up the super class as one that takes `num_graph_fn_inputs` inputs in its computation and
        # produces 1 output.
        super(ConcatLayer, self).__init__(scope=scope, num_graph_fn_inputs=num_graph_fn_inputs,
                                          num_graph_fn_outputs=1, **kwargs)
        self.axis = axis

    def create_variables(self, input_spaces):
        super(ConcatLayer, self).create_variables(input_spaces)

        # Wrapper for backend.
        if backend == "tf":
            import tensorflow as tf
            self.layer = tf.keras.layers.Concatenate(axis=self.axis)

    def _graph_fn_apply(self, *inputs):
        return self.layer.apply(force_list(inputs))

