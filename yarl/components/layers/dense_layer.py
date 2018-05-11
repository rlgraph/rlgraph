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
from yarl.utils.util import get_shape
from yarl.spaces import Space, Dict, Tuple
from .layer_component import LayerComponent
from .initializer import Initializer


class DenseLayer(LayerComponent):
    """
    A dense (or "fully connected") NN-layer.
    """
    def __init__(self, units, input_space, *sub_components, **kwargs):
        """
        Args:
            units (int): The number of nodes in this layer.
            input_space (Space): The input Space that will be passed through this layer.
                TODO: Lift restriction that this has to be primitive Space? Would this ever be needed?

        Keyword Args:
            weights_spec (any): A specifier for a weights initializer.
            biases_spec (any): A specifier for a biases initializer.
        """
        assert not isinstance(input_space, (Dict, Tuple)), "ERROR: Cannot handle container input_spaces atm!"
        super(DenseLayer, self).__init__(*sub_components, **kwargs)

        self.units = units

        # Create weights (and maybe biases).
        self.weights_shape = (input_space.shape[1], self.units)
        self.weights_spec = kwargs.get("weights_spec")
        self.weights_init = Initializer.from_spec(self.weights_spec, shape=self.weights_shape)
        self.biases_shape = (self.units,)
        self.biases_spec = kwargs.get("biases_spec")
        self.biases_init = Initializer.from_spec(self.biases_spec, shape=self.biases_shape)

        # Wrapper for backend.
        if backend() == "tf":
            import tensorflow as tf
            self.layer = tf.layers.Dense(units=self.units,
                                         kernel_initializer=self.weights_init.initializer,
                                         use_bias=(self.biases_init.initializer is not None),
                                         bias_initializer=self.biases_init.initializer)

    def _computation_apply(self, input_):
        """
        The actual calculation on a single, primitive input Space.

        Args:
            input_ (any): The input to the fc-layer.

        Returns:
            The output after having pushed the input through the layer.
        """
        return self.layer.apply(input_)
