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

from rlgraph.components.layers.nn.dense_layer import DenseLayer
from rlgraph.components.neural_networks.neural_network import NeuralNetwork
from rlgraph.utils.decorators import rlgraph_api
from rlgraph.utils.rlgraph_errors import RLGraphError


class ValueFunction(NeuralNetwork):
    """
    A Value-function is a wrapper Component that contains a NeuralNetwork and adds a value-function output.
    """
    def __init__(self, *layers, **kwargs):
        """
        Args:
            See `NeuralNetwork` Component.

        Keyword Args:
            See `NeuralNetwork` Component.
        """
        # Construct this NN.
        super(ValueFunction, self).__init__(*layers, scope=kwargs.pop("scope", "value-function"), **kwargs)
        # Add single-node value layer.
        super(ValueFunction, self).add_layer(DenseLayer(units=1, scope="value-function-output"))

    def add_layer(self, layer_component):
        raise RLGraphError("Cannot add a Layer to a completed ValueFunction object.")

    # OBSOLETED API method. Use `call` instead.
    @rlgraph_api
    def value_output(self, *inputs):
        self.logger.warning("In ValueFunction component: API `value_output` is obsoleted, use `call` instead!")
        return self.call(*inputs)
