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
from yarl.utils.util import get_rank
from yarl.components.loss_functions import LossFunction
from yarl.spaces import IntBox, sanity_check_space

if get_backend() == "tf":
    import tensorflow as tf


class DQNLossFunction(LossFunction):
    """
    The classic 2015 DQN Loss Function:
    L = Expectation-over-uniform-batch(r + gamma x max_a'Qt(s',a') - Qn(s,a))^2
    Where Qn is the "normal" Q-network and Qt is the "target" net (which is a little behind Qn for stability purposes).

    API:
        loss_per_item(q_values_s, actions, rewards, terminals, qt_values_sp, q_values_sp=None): The DQN loss per batch
            item.
    """
    def __init__(self, double_q=False, scope="dqn-loss-function", **kwargs):
        """
        Args:
            double_q (bool): Whether to use the double DQN loss function (see DQNAgent [2]).
        """
        self.double_q = double_q

        ## Pass our in-Socket names to parent c'tor.
        #input_sockets = ["q_values", "actions", "rewards", "terminals", "qt_values_s_"]
        ## For double-Q, we need an additional input for the q-net's s'-q-values (not the target's ones!).
        #if self.double_q:
        #    input_sockets.append("q_values_s_")

        super(DQNLossFunction, self).__init__(scope=scope, **kwargs)

        self.action_space = None
        self.ranks_to_reduce = 0  # How many ranks do we have to reduce to get down to the final loss per batch item?

    def check_input_spaces(self, input_spaces, action_space):
        """
        Do some sanity checking on the incoming Spaces:
        """
        self.action_space = action_space
        # Check for IntBox and num_categories.
        sanity_check_space(
            self.action_space, allowed_types=[IntBox], must_have_categories=True
        )
        self.ranks_to_reduce = len(self.action_space.get_shape(with_batch_rank=True)) - 1

    def _graph_fn_loss_per_item(self, q_values_s, actions, rewards, terminals, qt_values_sp, q_values_sp=None):
        """
        Args:
            q_values_s (SingleDataOp): The batch of Q-values representing the expected accumulated discounted returns
                when in s and taking different actions a.
            actions (SingleDataOp): The batch of actions that were actually taken in states s (from a memory).
            rewards (SingleDataOp): The batch of rewards that we received after having taken a in s (from a memory).
            terminals (SingleDataOp): The batch of terminal signals that we received after having taken a in s
                (from a memory).
            qt_values_sp (SingleDataOp): The batch of Q-values representing the expected accumulated discounted
                returns (estimated by the target net) when in s' and taking different actions a'.
            q_values_sp (Optional[SingleDataOp]): If `self.double_q` is True: The batch of Q-values representing the
                expected accumulated discounted returns (estimated by the (main) policy net) when in s' and taking
                different actions a'.

        Returns:
            SingleDataOp: The loss values vector (one single value for each batch item).
        """
        if get_backend() == "tf":
            if self.double_q:
                # For double-Q, we no longer use the max(a')Qt(s'a') value.
                # Instead, the a' used to get the Qt(s'a') is given by argmax(a') Q(s',a') <- Q=q-net, not target net!
                a_primes = tf.argmax(input=q_values_sp, axis=-1)

                # Now lookup Q(s'a') with the calculated a'.
                one_hot = tf.one_hot(indices=a_primes, depth=self.action_space.num_categories)
                qt_sp_ap_values = tf.reduce_sum(input_tensor=(qt_values_sp * one_hot), axis=-1)
            else:
                # Qt(s',a') -> Use the max(a') value (from the target network).
                qt_sp_ap_values = tf.reduce_max(input_tensor=qt_values_sp, axis=-1)

            # Make sure the rewards vector (batch) is broadcast correctly.
            for _ in range(get_rank(qt_sp_ap_values) - 1):
                rewards = tf.expand_dims(rewards, axis=1)

            # Ignore Q(s'a') values if s' is a terminal state. Instead use 0.0 as the state-action value for s'a'.
            # Note that in that case, the next_state (s') is not the correct next state and should be disregarded.
            # See Chapter 3.4 in "RL - An Introduction" (2017 draft) by A. Barto and R. Sutton for a detailed analysis.
            qt_sp_ap_values = tf.where(condition=terminals,
                                       x=tf.zeros_like(qt_sp_ap_values),
                                       y=qt_sp_ap_values)

            # Q(s,a) -> Use the Q-value of the action actually taken before.
            one_hot = tf.one_hot(indices=actions, depth=self.action_space.num_categories)
            q_s_a_values = tf.reduce_sum(input_tensor=(q_values_s * one_hot), axis=-1)

            # Calculate the TD-delta (target - current estimate).
            td_delta = (rewards + self.discount * qt_sp_ap_values) - q_s_a_values

            # Reduce over the composite actions, if any.
            if get_rank(td_delta) > 1:
                td_delta = tf.reduce_mean(input_tensor=td_delta, axis=list(range(1, self.ranks_to_reduce + 1)))

            return tf.pow(x=td_delta, y=2)

