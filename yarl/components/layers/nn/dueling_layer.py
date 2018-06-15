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
from yarl.components.layers.nn.nn_layer import NNLayer


if get_backend() == "tf":
    import tensorflow as tf


class DuelingLayer(NNLayer):
    """
    A dueling layer that separates a value function (per state) and an advantage (per action) stream, and then
    merges them again into a state-action value layer according to:
    Q = V + A - [mean(A)] (for details, see Equation (9) in: [1])

    [1] Dueling Network Architectures for Deep Reinforcement Learning, Wang et al. - 2016
    """
    def _graph_fn_apply(self, input_):
        """
        Args:
            input_ (SingleDataOp): The outputs from a previous Layer.

        Returns:
            SingleDataOp: The final calculated Q values based on: Q = V + [A - mean(A)]
        """
        # Use the very first node as value function output.
        # Use all following nodes as advantage function output.
        value_, advantages = tf.split(input_, (1, -1), axis=-1)

        if get_backend() == "tf":
            mean_advantage = tf.reduce_mean(advantages, axis=-1)
            return value_ + advantages - mean_advantage
