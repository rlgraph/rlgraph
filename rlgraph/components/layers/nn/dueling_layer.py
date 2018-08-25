# Copyright 2018 The RLgraph authors. All Rights Reserved.
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
from rlgraph.utils import RLGraphError
from rlgraph.components.layers.nn.nn_layer import NNLayer
from rlgraph.utils.util import get_rank


if get_backend() == "tf":
    import tensorflow as tf


class DuelingLayer(NNLayer):
    """
    A dueling layer that separates a value function (per state) and an advantage (per action) stream, and then
    merges them again into a state-action value layer according to:
    Q = V + A - [mean(A)] (for details, see Equation (9) in: [1])

    [1] Dueling Network Architectures for Deep Reinforcement Learning, Wang et al. - 2016

    API:
        apply(input_) -> state_value, advantage_values, q_values
    """
    def __init__(self, scope="dueling-layer", **kwargs):
        raise RLGraphError("DuelingLayer has been obsoleted for the time being. "
                           "Use dueling-action-adapter directly.")
        super(DuelingLayer, self).__init__(scope=scope, **kwargs)

        self.num_advantage_values = None
        self.target_space = None

    def check_input_spaces(self, input_spaces, action_space=None):
        super(DuelingLayer, self).check_input_spaces(input_spaces, action_space)
        in_space = input_spaces["inputs"]
        # Last rank is the [value + advantage-values] rank, store the number of advantage values here.
        self.num_advantage_values = in_space.shape[-1] - 1

        self.target_space = action_space.with_batch_rank()

    def _graph_fn_apply(self, inputs):
        """
        Args:
            inputs (SingleDataOp): The flattened inputs to this layer. These must include the single node
                for the state-value.

        Returns:
            Tuple[SingleDataOp]: The calculated, reshaped Q values (for each composite action) based on: Q = V + [A - mean(A)]
                - state_value (SingleDataOp): The single state-value (not dependent on actions).
                - advantage_values (SingleDataOp): The already reshaped advantage values per action.
                - q_values (SingleDataOp): The already reshaped state-action (q) values per action.
        """
        # Use the very first node as value function output.
        # Use all following nodes as advantage function output.
        if get_backend() == "tf":
            # Separate out the single state-value node.
            state_value, advantages = tf.split(
                value=inputs, num_or_size_splits=(1, self.num_advantage_values), axis=-1
            )
            # Now we have to reshape the advantages according to our action space.
            shape = list(self.target_space.get_shape(with_batch_rank=-1, with_category_rank=True))
            advantages = tf.reshape(tensor=advantages, shape=shape)
            # Calculate the q-values according to [1] and return.
            mean_advantage = tf.reduce_mean(input_tensor=advantages, axis=-1, keepdims=True)

            # Make sure we broadcast the state_value correctly for the upcoming q_value calculation.
            state_value_expanded = state_value
            for _ in range(get_rank(advantages) - 2):
                state_value_expanded = tf.expand_dims(state_value_expanded, axis=1)
            q_values = state_value_expanded + advantages - mean_advantage

            # state-value, advantages, q_values
            return tf.squeeze(state_value, axis=-1), advantages, q_values
