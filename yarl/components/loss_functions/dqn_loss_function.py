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

from yarl import backend
from yarl.components.loss_functions import LossFunction


class DQNLossFunction(LossFunction):
    """
    The classic 2015 DQN Loss Function:
    L = Expectation-over-uniform-batch(r + gamma x maxa'Qt(s',a') - Qn(s,a))²
    Where Qn is the "normal" Q-network and Qt is the "target" net (which is a little behind Qn for stability purposes).
    """
    def __init__(self, discount=0.98, double_q=False, scope="dqn-loss-function", **kwargs):
        """
        Args:
            discount (float): The discount factor (gamma).
            double_q (bool): Whether to use the double DQN loss function (see DQNAgent [2]).
        """
        # Pass our in-Socket names to parent c'tor.
        super(DQNLossFunction, self).__init__("q_values", "actions", "rewards", "q_values_s_", scope=scope, **kwargs)
        self.discount = discount  # TODO: maybe move this to parent?
        self.double_q = double_q

    def check_input_spaces(self, input_spaces):
        """
        Do some sanity checking on the incoming Spaces:
        """
        # Loop through all our in-Sockets and sanity check each one of them.
        for in_sock in self.input_sockets:
            in_space = input_spaces[in_sock.name]
            # All input Spaces need batch ranks.
            assert in_space.has_batch_rank, "ERROR: Space in Socket '{}' to DQNLossFunction must have a " \
                                            "batch rank (0th position)!".format(in_sock.name, self.name)

    def _graph_fn_loss_per_item(self, q_values, actions, rewards, q_values_s_):
        """
        Args:
            q_values (DataOp): The Q-values representing the expected accumulated discounted returns when in s and
                taking different actions a.
            actions (DataOp): The actions that were actually taken in states s (from a memory).
            rewards (DataOp): The rewards that we received after having taken a in s (from a memory).
            q_values_s_ (DataOp): The Q-values representing the expected accumulated discounted returns when in s'
                and taking different actions a'.

        Returns:
            DataOp: The average loss value of the batch.
        """
        if backend == "tf":
            import tensorflow as tf

            batch_size = tf.shape(q_values)[0]

            # q_values are reduced by index using gather_nd, but keep in mind it could be done as well
            # via one_hot->matmul->reduce_sum.

            # TODO: What if action space is complex? Pretend that q_values are already re-shaped according to action-Space.
            # Construct gather indexes from actions.
            batch_indexes = tf.expand_dims(tf.range(start=0, limit=batch_size), axis=1)
            gather_indexes = tf.concat([batch_indexes] + actions, axis=1)
            # Calculate the TD-delta (target - current estimate).
            td_delta = (rewards + self.discount * tf.reduce_max(q_values_s_, axis=-1)) - \
                       tf.gather_nd(q_values, gather_indexes)
            return tf.pow(td_delta, 2)
