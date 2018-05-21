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


class DenseLayer(NNLayer):
    """
    A dense (or "fully connected") NN-layer.
    """
    def __init__(self, units, scope="dense-layer", *sub_components, **kwargs):
        """
        Args:
            units (int): The number of nodes in this layer.

        Keyword Args:
            weights_spec (any): A specifier for a weights initializer.
            biases_spec (any): A specifier for a biases initializer. If False, use no biases.
        """
        # Remove kwargs before calling super().
        self.weights_spec = kwargs.pop("weights_spec", None)
        self.biases_spec = kwargs.pop("biases_spec", False)

        super(DenseLayer, self).__init__(*sub_components, scope=scope, **kwargs)

        self.weights_init = None  # at build time
        self.biases_init = None  # at build time

        # Number of nodes in this layer.
        self.units = units

    def create_variables(self, input_spaces):
        super(DenseLayer, self).create_variables(input_spaces)

        in_space = input_spaces["input"]

        # Create weights.
        weights_shape = (in_space.shape[0], self.units)  # [0] b/c `in_space.shape` does not include batch-rank
        self.weights_init = Initializer.from_spec(shape=weights_shape, specification=self.weights_spec)
        # And maybe biases.
        biases_shape = (self.units,)
        self.biases_init = Initializer.from_spec(shape=biases_shape, specification=self.biases_spec)

        # Wrapper for backend.
        if backend() == "tf":
            import tensorflow as tf
            # TODO: variables registry (variables now exist in tf.layer).
            self.layer = tf.layers.Dense(units=self.units,
                                         kernel_initializer=self.weights_init.initializer,
                                         use_bias=(self.biases_spec is not False),
                                         bias_initializer=self.biases_init.initializer)

