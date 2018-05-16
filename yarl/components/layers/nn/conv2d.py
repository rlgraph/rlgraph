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

from .initializer import Initializer
from .nn_layer import NNLayer


class Conv2DLayer(NNLayer):
    """
    A Conv2D NN-layer.
    """
    def __init__(self, filters, kernel_size, strides, *sub_components, **kwargs):
        """
        Args:
            filters (int): The number of filters to produce in the channel-rank.
            kernel_size (Union[int,Tuple[int]]): The height and width (or one value for both) of the 2D convolution
                sliding window.
            strides (Union[int,Tuple[int]]): Kernel stride size along height and width axis (or one value
                for both directions).

        Keyword Args:
            padding (str): One of 'valid' or 'same'. Default: 'valid'.
            data_format (str): One of 'channels_last' (default) or 'channels_first'. Specifies which rank (first or
                last) is the color-channel. If the input Space is with batch, the batch always has the first rank.
            activation (Optional[op]): The activation function to use. Default: None.
            kernel_spec (any): A specifier for the kernel-weights initializer.
            bias_spec (any): A specifier for the biases-weights initializer. If False, use no biases.
            # TODO: regularization specs
        """
        super(Conv2DLayer, self).__init__(*sub_components, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)
        self.padding = kwargs.get("padding")
        self.data_format = kwargs.get("data_format")
        self.activation = kwargs.get("activation")
        self.kernel_spec = kwargs.get("kernel_spec")
        self.kernel_init = None  # at model-build time
        self.biases_spec = kwargs.get("biases_spec", False)
        self.biases_init = None  # at model-build time

    def create_variables(self, input_spaces):
        super(Conv2DLayer, self).create_variables(input_spaces)

        # Create kernel and biases initializers.
        self.kernel_init = Initializer.from_spec(shape=self.kernel_size, specification=self.kernel_spec)
        self.biases_init = Initializer.from_spec(shape=self.kernel_size, specification=self.biases_spec)

        # Wrapper for backend.
        if backend() == "tf":
            import tensorflow as tf
            # TODO: variables registry (variables now exist in tf.layer).
            self.layer = tf.layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size,
                                          strides=self.strides, padding=self.padding,
                                          data_format=self.data_format,
                                          activation=self.activation,
                                          use_bias=(self.biases_spec is not False),
                                          kernel_initializer=self.kernel_init,
                                          bias_initializer=self.biases_init
                                          )
