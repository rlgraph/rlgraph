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

from yarl import get_backend
from yarl.spaces import sanity_check_space
from yarl.components.layers.nn.nn_layer import NNLayer


if get_backend() == "tf":
    import tensorflow as tf


class DuelingLayer(NNLayer):
    """
    A dueling layer that separates a value function (per state) and an advantage (per action) stream, and then
    merges them again into a state-action value layer according to:
    Q = V + A - [mean(A)] (for details, see Equation (9) in: [1])

    [1] Dueling Network Architectures for Deep Reinforcement Learning, Wang et al. - 2016

    API:
    ins:
        input (SingleDataOp): The flattened input to the dueling layer. Its number of nodes corresponds to:
            Flattened action-space + 1 (state-value).
    outs:
        state_value (SingleDataOp): The single state-value (not dependent on actions).
        advantage_values (SingleDataOp): The already reshaped advantage values per action.
        q_values (SingleDataOp): The already reshaped state-action (q) values per action.
    """
    def __init__(self, scope="dueling-layer", **kwargs):
        # We have 3 out-Sockets for our apply graph_fn.
        super(DuelingLayer, self).__init__(scope=scope, num_graph_fn_outputs=3, **kwargs)

        # Define our interface.
        # Rename output sockets into proper names.
        self.rename_socket("output0", "state_value")
        self.rename_socket("output1", "advantage_values")
        self.rename_socket("output2", "q_values")
        # Add a mirrored "output" (q_values) for clarity.
        self.define_outputs("output")
        self.connect("q_values", "output")

        self.num_advantage_values = None
        self.target_space = None

    def check_input_spaces(self, input_spaces, action_space):
        super(DuelingLayer, self).check_input_spaces(input_spaces, action_space)
        in_space = input_spaces["input"]
        # Last rank is the [value + advantage-values] rank, store the number of advantage values here.
        self.num_advantage_values = in_space.get_shape(with_batch_rank=True)[-1] - 1

        self.target_space = action_space.with_batch_rank()

    def _graph_fn_apply(self, flat_input):
        """
        Args:
            flat_input (SingleDataOp): The flattened inputs to this layer. These must include the single node for the
                state-value.

        Returns:
            SingleDataOp: The calculated, reshaped Q values (for each composite action) based on: Q = V + [A - mean(A)]
        """
        # Use the very first node as value function output.
        # Use all following nodes as advantage function output.
        if get_backend() == "tf":
            # Separate out the single state-value node.
            state_value, advantages = tf.split(
                value=flat_input, num_or_size_splits=(1, self.num_advantage_values), axis=-1
            )
            state_value = tf.squeeze(input=state_value, axis=-1)
            # Now we have to reshape the advantages according to our action space.
            shape = list(self.target_space.get_shape(with_batch_rank=-1, with_category_rank=True))
            advantages = tf.reshape(tensor=advantages, shape=shape)
            # Calculate the q-values according to [1] and return.
            mean_advantage = tf.reduce_mean(input_tensor=advantages, axis=-1, keepdims=True)
            # state-value, advantages, q_values
            return state_value, advantages, state_value + advantages - mean_advantage
