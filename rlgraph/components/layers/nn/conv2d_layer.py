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
from rlgraph.utils import PyTorchVariable
from rlgraph.utils.initializer import Initializer
from rlgraph.components.layers.nn.nn_layer import NNLayer
from rlgraph.components.layers.nn.activation_functions import get_activation_function

if get_backend() == "tf":
    import tensorflow as tf
elif get_backend() == "pytorch":
    import torch.nn as nn
    from rlgraph.utils.pytorch_util import get_input_channels, SamePaddedConv2d


class Conv2DLayer(NNLayer):
    """
    A Conv2D NN-layer.
    """
    def __init__(self, filters, kernel_size, strides, padding="valid", data_format="channels_last",
                 kernel_spec=None, biases_spec=None, **kwargs):
        """
        Args:
            filters (int): The number of filters to produce in the channel-rank.
            kernel_size (Union[int,Tuple[int]]): The height and width (or one value for both) of the 2D convolution
                sliding window.
            strides (Union[int,Tuple[int]]): Kernel stride size along height and width axis (or one value
                for both directions).
            padding (str): One of 'valid' or 'same'. Default: 'valid'.
            data_format (str): One of 'channels_last' (default) or 'channels_first'. Specifies which rank (first or
                last) is the color-channel. If the input Space is with batch, the batch always has the first rank.
            kernel_spec (any): A specifier for the kernel-weights initializer. Use None for the default initializer.
                Default: None.
            bias_spec (any): A specifier for the biases-weights initializer. Use None for the default initializer.
                If False, uses no biases. Default: False.

            # TODO: regularization specs
        """
        super(Conv2DLayer, self).__init__(scope=kwargs.pop("scope", "conv-2d"), **kwargs)

        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size)
        self.strides = strides if isinstance(strides, (tuple, list)) else (strides, strides)
        self.padding = padding
        self.data_format = data_format
        self.kernel_spec = kernel_spec
        self.biases_spec = biases_spec

        # At model-build time.
        self.kernel_init = None
        # At model-build time.
        self.biases_init = None

    def create_variables(self, input_spaces, action_space=None):
        in_space = input_spaces["inputs[0]"]

        # Create kernel and biases initializers.
        self.kernel_init = Initializer.from_spec(shape=self.kernel_size, specification=self.kernel_spec)
        self.biases_init = Initializer.from_spec(shape=self.kernel_size, specification=self.biases_spec)

        # Wrapper for backend.
        if get_backend() == "tf":
            self.layer = tf.layers.Conv2D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                activation=get_activation_function(self.activation, *self.activation_params),
                use_bias=(self.biases_spec is not False),
                kernel_initializer=self.kernel_init.initializer,
                bias_initializer=(self.biases_init.initializer or tf.zeros_initializer()),
                trainable=(False if self.trainable is False else True)
            )

            # Now build the layer so that its variables get created.
            self.layer.build(in_space.get_shape(with_batch_rank=True))
            # Register the generated variables with our registry.
            self.register_variables(*self.layer.variables)
        elif get_backend() == "pytorch":
            shape = in_space.shape
            num_channels = get_input_channels(shape)
            apply_bias = (self.biases_spec is not False)

            # print("Defining conv2d layer with shape = {} and channels {}".format(
            #     shape, num_channels
            # ))
            if self.padding == "same":
                # N.b. there is no 'same' or 'valid' padding for PyTorch so need custom layer.
                self.layer = SamePaddedConv2d(
                    in_channels=num_channels,
                    out_channels=self.filters,
                    # Only support square kernels.
                    kernel_size=self.kernel_size[0],
                    stride=self.strides,
                    bias=apply_bias
                )
            else:
                self.layer = nn.Conv2d(
                    in_channels=num_channels,
                    out_channels=self.filters,
                    kernel_size=self.kernel_size,
                    stride=self.strides,
                    padding=0,
                    bias=apply_bias
                )
            # Apply weight initializer
            if self.kernel_init.initializer is not None:
                # Must be a callable in PyTorch
                self.kernel_init.initializer(self.layer.weight)
            if apply_bias:
                if self.biases_spec is not None and self.biases_init.initializer is not None:
                    self.biases_init.initializer(self.layer.bias)
                else:
                    # Fill with zeros.
                    self.layer.bias.data.fill_(0)
            if self.activation is not None:
                # Activation function will be used in apply.
                self.activation_fn = get_activation_function(self.activation, *self.activation_params)
            self.register_variables(PyTorchVariable(name=self.global_scope, ref=self.layer))

