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

from rlgraph.components.layers import ConcatLayer
from rlgraph.components.neural_networks.neural_network import NeuralNetwork
from rlgraph.components.neural_networks.value_function import ValueFunction
from rlgraph.utils.decorators import rlgraph_api


class QFunction(ValueFunction):
    """
    A Q-function network taking state and action inputs and concatenating them either at the very beginning OR -
    if an extra image_stack_spec is given - only after passing the states through this image_stack.
    """
    def __init__(self, *layers, **kwargs):
        self.image_stack = None
        image_stack_spec = kwargs.pop("image_stack_spec", None)
        if image_stack_spec is not None:
            self.image_stack = NeuralNetwork.from_spec(image_stack_spec)
        main_nn_spec = kwargs.pop("network_spec", None)
        if main_nn_spec is not None:
            self.main_nn = NeuralNetwork.from_spec(main_nn_spec)
        else:
            self.main_nn = NeuralNetwork(*layers, scope="main-nn", **kwargs)

        self.concat_layer = ConcatLayer()
        sub_components = ([self.image_stack] if self.image_stack is not None else []) + \
                         [self.concat_layer, self.main_nn]
        super(QFunction, self).__init__(*sub_components, num_inputs=2, scope=kwargs.pop("scope", "q-function"), **kwargs)

    @rlgraph_api
    def call(self, states, actions):
        """
        Computes Q(s,a) by passing states and actions through one or multiple processing stacks..
        """
        # Send states through separate stack, only then concat.
        if self.image_stack is not None:
            image_processing_output = self.image_stack.call(states)
            concat_input = (image_processing_output, actions)
        # Concat states and actions, then pass through.
        else:
            concat_input = (states, actions)

        states_and_actions = self.concat_layer.call(concat_input)
        return self.main_nn.call(states_and_actions)
