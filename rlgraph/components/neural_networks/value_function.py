# Copyright 2018/2019 The RLgraph authors. All Rights Reserved.
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

from __future__ import absolute_import, division, print_function

from rlgraph.components.component import Component
from rlgraph.components.layers.layer import Layer
from rlgraph.components.neural_networks.neural_network import NeuralNetwork
from rlgraph.utils.decorators import rlgraph_api


class ValueFunction(Component):
    """
    A Value-function is a wrapper Component that contains a NeuralNetwork and adds a value-function output.
    """
    def __init__(self, network_spec, scope="value-function", **kwargs):
        """
        Args:
            network_spec (list): Layer specification for baseline network.
        """
        super(ValueFunction, self).__init__(scope=scope, **kwargs)
        self.network_spec = network_spec

        # If first layer is conv, build image stack.
        # TODO: This should not be checked in this way. What if network_spec if a NN Component, a dict, etc..?
        self.use_image_stack = False
        if isinstance(self.network_spec, (list, tuple)) and self.network_spec[0]["type"] == "conv2d":
            self.use_image_stack = True
        self.image_stack = None
        self.dense_stack = None

        self.neural_network = None
        self.value_layer_spec = {
            "type": "dense",
            "units": 1,
            "activation": "linear",
            "scope": "value-function-output"
        }
        self.build_value_function()

    def build_value_function(self):
        # Attach VF output to hidden layers.
        self.neural_network = NeuralNetwork.from_spec(self.network_spec)
        self.neural_network.add_layer(Layer.from_spec(self.value_layer_spec))
        self.add_components(self.neural_network)

    @rlgraph_api
    def value_output(self, nn_inputs):
        """
        Args:
            nn_inputs (any): The inputs to our neural network.

        Returns:
            any: Value function estimate V(s) for inputs s.
        """
        return self.neural_network.call(nn_inputs)
