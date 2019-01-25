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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from rlgraph.components import Component
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

        # Attach VF output to hidden layers.
        value_layer = {
            "type": "dense",
            "units": 1,
            "activation": "linear",
            "scope": "value-function-output"
        }
        network_spec.append(value_layer)
        self.neural_network = NeuralNetwork.from_spec(network_spec)
        self.add_components(self.neural_network)

    @rlgraph_api
    def value_output(self, nn_input, internal_states=None):
        """
        Args:
            nn_input (any): The input to our neural network.
            internal_states (Optional[any]): The initial internal states going into an RNN-based neural network.

        Returns:
            any: Value function estimate V(s) for inputs s.
        """
        nn_output = self.neural_network.apply(nn_input, internal_states)
        return nn_output["output"]
