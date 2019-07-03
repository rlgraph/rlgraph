# Copyright 2018/2019 The RLgraph authors, All Rights Reserved.
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

from rlgraph.components.layers import Layer, ConcatLayer
from rlgraph.components.neural_networks.neural_network import NeuralNetwork
from rlgraph.components.neural_networks.value_function import ValueFunction
from rlgraph.utils.decorators import rlgraph_api


class QFunction(ValueFunction):
    """
    A Q-function network taking state and action inputs.
    """
    def __init__(self, scope="q-function", **kwargs):
        self.image_stack = None
        self.dense_stack = None

        super(QFunction, self).__init__(scope=scope, **kwargs)

        # Add all sub-components to this one.
        #if self.image_stack is not None:
        #    self.add_components(self.image_stack)
        self.concat_layer = ConcatLayer()
        #self.add_components(self.concat_layer, self.dense_stack)

    def build_value_function(self):
        """
        Builds a dense stack and optionally an image stack.
        """
        if self.image_stack is not None:
            image_components = []
            dense_components = []
            for layer_spec in self.network_spec:
                if layer_spec["type"] in ["conv2d", "reshape"]:
                    image_components.append(Layer.from_spec(layer_spec))

            self.image_stack = NeuralNetwork(*image_components, scope="image-stack")

            # Remaining layers should be dense.
            for layer_spec in self.network_spec[len(image_components):]:
                assert layer_spec["type"] == "dense", "Only expecting dense layers after image " \
                                                      "stack but found spec: {}.".format(layer_spec)
                dense_components.append(layer_spec)

            dense_components.append(self.value_layer_spec)
            self.dense_stack = NeuralNetwork(*dense_components, scope="dense-stack")
            self.neural_network = NeuralNetwork(self.image_stack, self.dense_stack)
        else:
            # Assume dense network otherwise -> only a single stack.
            dense_components = []
            for layer_spec in self.network_spec:
                assert layer_spec["type"] == "dense", \
                    "ERROR: Only dense layers allowed if not using image stack in this network."
                dense_components.append(Layer.from_spec(layer_spec))
            dense_components.append(self.value_layer_spec)
            self.dense_stack = NeuralNetwork(*dense_components, scope="dense-stack")

    @rlgraph_api
    def state_action_value(self, states, actions, internal_states=None):
        """
        Computes Q(s,a) by passing states and actions through one or multiple processing stacks..
        """
        if self.image_stack is not None:
            image_processing_output = self.image_stack.call(states)
            state_actions = self.concat_layer.call(image_processing_output, actions)
            dense_output = self.dense_stack.call(state_actions)
        else:
            # Concat states and actions, then pass through.
            state_actions = self.concat_layer.call(states, actions)
            dense_output = self.dense_stack.call(state_actions)

        return dense_output
