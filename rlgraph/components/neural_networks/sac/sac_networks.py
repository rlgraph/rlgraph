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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from rlgraph.components.layers.nn.concat_layer import ConcatLayer
from rlgraph.components.neural_networks.neural_network import NeuralNetwork
from rlgraph.utils.decorators import rlgraph_api


class SACValueNetwork(NeuralNetwork):
    """
    Value network for SAC which must be able to merge different input types.
    """
    def __init__(self, network_spec, use_image_stack, scope="sac-value-network", **kwargs):
        """
        Args:
            network_spec (dict): Network spec.
            use_image_stack (bool): If true, build image stack, then concatenate output of image stack with
                actions to compute Q(s,a).
        """
        super(SACValueNetwork, self).__init__(scope=scope, **kwargs)

        self.network_spec = network_spec
        self.use_image_stack = use_image_stack

        self.image_stack = None
        self.dense_stack = None
        self.build_stacks()

        # The concatenation layer.
        self.concat_layer = ConcatLayer()

        # Add all sub-components to this one.
        if self.image_stack is not None:
            self.add_components(self.image_stack)

        self.add_components(self.dense_stack, self.concat_layer)

    @staticmethod
    def build_stacks():
        """
        Builds a dense stack and optionally an image stack.
        """
        # sub_components = []
        #
        # # Divide by 255
        #
        # for i, (num_filters, kernel_size, stride) in enumerate(zip([16, 32], [8, 4], [4, 2])):
        #     # Conv2D plus ReLU activation function.
        #     conv2d = Conv2DLayer(
        #         filters=num_filters, kernel_size=kernel_size, strides=stride, padding="same",
        #         activation="relu", scope="conv2d-{}".format(i)
        #     )
        #     sub_components.append(conv2d)
        #
        # # A Flatten preprocessor and then an fc block (surrounded by ReLUs) and a time-rank-unfolding.
        # sub_components.extend([
        #     ReShape(flatten=True, scope="flatten"),  # Flattener (to flatten Conv2D output for the fc layer).
        #     DenseLayer(units=256),  # Dense layer.
        #     NNLayer(activation="relu", scope="relu-before-lstm"),
        # ])
        #
        # #stack_before_unfold = <- formerly known as
        # image_stack = Stack(sub_components, scope="image-stack")

    @rlgraph_api
    def apply(self, states, actions):
        """
        Computes Q(s,a) by passing states and actions through one or multiple processing stacks.
        """

        image_processing_output = self.image_stack.apply(states)

        # Concat everything together.
        concatenated_data = self.concat_layer.apply(
            image_processing_output,  actions
        )

        dense_output = self.dense_stack.apply(concatenated_data)

        return dense_output
