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


class DenseLayer(NNLayer):
    """
    A dense (or "fully connected") NN-layer.
    """
    def __init__(self, units, weights_spec=None, biases_spec=None, **kwargs):
        """
        Args:
            units (int): The number of nodes in this layer.
            weights_spec (any): A specifier for a weights initializer. If None, use the default initializer.
            biases_spec (any): A specifier for a biases initializer. If False, use no biases. If None,
                use the default initializer (0.0).
        """
        super(DenseLayer, self).__init__(scope=kwargs.pop("scope", "dense-layer"), **kwargs)

        self.weights_spec = weights_spec
        self.biases_spec = biases_spec
        # At build time.
        self.weights_init = None
        self.biases_init = None

        # Number of nodes in this layer.
        self.units = units

    def create_variables(self, input_spaces, action_space=None):
        in_space = input_spaces["inputs[0]"]
        assert in_space.rank > 0, "ERROR: Must have input Space ({}) with rank larger 0!".format(in_space)

        # Create weights matrix and (maybe) biases vector.
        weights_shape = (in_space.shape[0], self.units)
        self.weights_init = Initializer.from_spec(shape=weights_shape, specification=self.weights_spec)
        biases_shape = (self.units,)
        self.biases_init = Initializer.from_spec(shape=biases_shape, specification=self.biases_spec)

        # Wrapper for backend.
        if get_backend() == "tf":
            self.layer = tf.layers.Dense(
                units=self.units,
                activation=get_activation_function(self.activation, *self.activation_params),
                kernel_initializer=self.weights_init.initializer,
                use_bias=(self.biases_spec is not False),
                bias_initializer=(self.biases_init.initializer or tf.zeros_initializer()),
                trainable=(False if self.trainable is False else True)
            )

            # Now build the layer so that its variables get created.
            self.layer.build(in_space.get_shape(with_batch_rank=True))
            # Register the generated variables with our registry.
            self.register_variables(*self.layer.variables)
        elif get_backend() == "pytorch":
            # N.b. activation must be added as a separate 'layer' when assembling a network.
            # In features is the num of input channels.
            apply_bias = (self.biases_spec is not False)
            in_features = in_space.shape[1] if in_space.shape[0] == 1 else in_space.shape[0]
            # print("name = {}, ndim = {}, in space.shape = {}, in_features = {}, units = {}".format(
            #     self.name, ndim, in_space.shape, in_features, self.units))
            self.layer = nn.Linear(
                # In case there is a batch dim here due to missing preprocessing.
                in_features=in_features,
                out_features=self.units,
                bias=apply_bias
            )
            # Apply weight initializer
            if self.weights_init.initializer is not None:
                # Must be a callable in PyTorch
                self.weights_init.initializer(self.layer.weight)
            if apply_bias:
                if self.biases_spec is not None and self.biases_init.initializer is not None:
                    self.biases_init.initializer(self.layer.bias)
                else:
                    # Fill with zeros.
                    self.layer.bias.data.fill_(0)
            if self.activation is not None:
                # Activation function will be used in apply.
                self.activation_fn = get_activation_function(self.activation, *self.activation_params)
            # Use unique scope as name.
            self.register_variables(PyTorchVariable(name=self.global_scope, ref=self.layer))
