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

from rlgraph import get_backend
from rlgraph.components.neural_networks.stack import Stack
from rlgraph.components.layers import Layer
from rlgraph.utils.decorators import rlgraph_api
from rlgraph.components.neural_networks.value_function import ValueFunction

if get_backend() == "tf":
    import tensorflow as tf
elif get_backend() == "pytorch":
    import torch


class SACValueNetwork(ValueFunction):
    """
    Value network for SAC which must be able to merge different input types.
    """
    def __init__(self, scope="sac-value-network", **kwargs):
        super(SACValueNetwork, self).__init__(scope=scope, **kwargs)
        self.image_stack = None
        self.dense_stack = None

        # If first layer is conv, build image stack.
        self.use_image_stack = self.network_spec[0]["type"] == "conv2d"
        self.build_value_function()

        # Add all sub-components to this one.
        if self.image_stack is not None:
            self.add_components(self.image_stack)

        self.add_components(self.dense_stack)

    def build_value_function(self):
        """
        Builds a dense stack and optionally an image stack.
        """
        if self.use_image_stack:
            image_components = []
            dense_components = []
            for layer_spec in self.network_spec:
                if layer_spec["type"] in ["conv2d", "reshape"]:
                    image_components.append(Layer.from_spec(layer_spec))

            self.image_stack = Stack(image_components, scope="image-stack")

            # Remainings layers should be dense.
            for layer_spec in self.network_spec[len(image_components):]:
                assert layer_spec["type"] == "dense", "Only expecting dense layers after image " \
                                                      "stack but found spec: {}.".format(layer_spec)
                dense_components.append(layer_spec)

            dense_components.append(self.value_layer )
            self.dense_stack = Stack(dense_components, scope="dense-stack")
        else:
            # Assume dense network otherwise -> onyl a single stack.
            dense_components = []
            for layer_spec in self.network_spec:
                assert layer_spec["type"] == "dense", "Only dense layers allowed if not using" \
                                                      " image stack in this network."
                dense_components.append(Layer.from_spec(layer_spec))
            dense_components.append(self.value_layer )
            self.dense_stack = Stack(dense_components, scope="dense-stack")

    @rlgraph_api
    def value_output(self,  nn_input, internal_states=None):
        """
        Computes Q(s,a) by passing states and actions through one or multiple processing stacks.

        Args:
            state_actions (list): Tuple containing state and flat actions.
        """
        states = nn_input[0]
        actions = nn_input[1:]
        concat_state_actions = None
        if self.use_image_stack:
            image_processing_output = self.image_stack.apply(states)

            # Concat everything together.
            if get_backend() == "tf":
                concat_state_actions = tf.concat([image_processing_output, actions], axis=-1)
            elif get_backend() == "pytorch":
                concat_state_actions = torch.cat([image_processing_output, actions], dim=-1)

            dense_output = self.dense_stack.apply(concat_state_actions)
        else:
            # Concat states and actions, then pass through.
            if get_backend() == "tf":
                concat_state_actions = tf.concat(nn_input, axis=-1)
            elif get_backend() == "pytorch":
                concat_state_actions = torch.cat(nn_input, dim=-1)
            dense_output = self.dense_stack.apply(concat_state_actions)

        return dense_output
