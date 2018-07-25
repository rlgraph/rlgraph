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
from rlgraph.utils.util import force_list

from .nn_layer import NNLayer

if get_backend() == "tf":
    import tensorflow as tf


class ConcatLayer(NNLayer):
    """
    A simple concatenation layer wrapper. The ConcatLayer is a Layer without sub-components but with n
    api_methods and 1 output, where the in-Sockets's data are concatenated into one out-Socket by its GraphFunction.
    """
    def __init__(self, axis=-1, scope="concat-layer", num_graph_fn_inputs=2, **kwargs):
        """
        Args:
            axis (int): The axis along which to concatenate. Use negative numbers to count from end.
                All api_methods to this layer must have the same shape, except for the `axis` rank.
                Default: -1.
        """
        super(ConcatLayer, self).__init__(scope=kwargs.pop("scope", "concat-layer"), **kwargs)
        self.axis = axis

    def check_input_spaces(self, input_spaces, action_space):
        super(ConcatLayer, self).check_input_spaces(input_spaces, action_space)
        in_space_0 = input_spaces["apply"][0]
        # Make sure all api_methods have the same shape except for the last rank.
        for i, in_space in enumerate(input_spaces["apply"]):
            assert in_space_0.shape[:-1] == in_space.shape[:-1], \
                "ERROR: Input spaces to ConcatLayer must have same shape except for last rank. {}st input's shape is {}, but " \
                "1st input's shape is {}.".format(i, in_space.shape, in_space_0.shape)

    def create_variables(self, input_spaces, action_space):
        super(ConcatLayer, self).create_variables(input_spaces, action_space)

        # Wrapper for backend.
        if get_backend() == "tf":
            self.layer = tf.keras.layers.Concatenate(axis=self.axis)

    def _graph_fn_apply(self, *inputs):
        return self.layer.apply(force_list(inputs))

