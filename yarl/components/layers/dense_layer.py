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
from yarl.spaces import Space, Dict, Tuple
from .layer_component import LayerComponent
from .initializer import Initializer


class DenseLayer(LayerComponent):
    """
    A dense (or "fully connected") NN-layer.
    """
    def __init__(self, units, *sub_components, **kwargs):
        """
        Args:
            units (int): The number of nodes in this layer.

        Keyword Args:
            weights_spec (any): A specifier for a weights initializer.
            biases_spec (any): A specifier for a biases initializer.
        """
        super(DenseLayer, self).__init__(*sub_components, **kwargs)
        self.weights_spec = kwargs.get("weights_spec")
        self.weights_init = None  # at build time
        self.biases_spec = kwargs.get("biases_spec", False)
        self.biases_init = None  # at build time

        # Number of nodes in this layer.
        self.units = units

        # The wrapped layer object.
        self.layer = None

    def create_variables(self):
        space = self.get_input().space
        assert not isinstance(space, (Dict, Tuple)), "ERROR: Cannot handle container input Spaces " \
                                                     "(atm; may soon do)!"
        assert space.rank > 1, \
            "ERROR: Rank of input-space (rank={}) must be 2 or larger (1st rank is batch size)!".\
                format(space.rank)

        # Create weights.
        weights_shape = (space.shape[1], self.units)
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

    def _computation_apply(self, input_):
        """
        The actual calculation on a single, primitive input Space.

        Args:
            input_ (any): The input to the fc-layer.

        Returns:
            The output after having pushed the input through the layer.
        """
        return self.layer.apply(input_)
