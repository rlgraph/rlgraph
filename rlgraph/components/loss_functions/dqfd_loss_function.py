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
from rlgraph.utils.decorators import rlgraph_api
from rlgraph.components.loss_functions.dqn_loss_function import DQNLossFunction
from rlgraph.utils.util import get_rank

if get_backend() == "tf":
    import tensorflow as tf


class DQFDLossFunction(DQNLossFunction):
    """
    The DQFD-loss extends the (dueling) DQN loss by a supervised loss to leverage expert demonstrations. Paper:

    https://arxiv.org/abs/1704.03732

    API:
        loss_per_item(q_values_s, actions, rewards, terminals, qt_values_sp, q_values_sp=None): The DQN loss per batch
            item.
    """
    def __init__(self, expert_margin=0.5, supervised_weight=1.0, scope="dqfd-loss-function", **kwargs):
        """
        Args:
            expert_margin (float): The expert margin enforces a distance in Q-values between expert action and
                all other actions.
            supervised_weight (float): Indicates weight of the expert loss.
        """
        self.expert_margin = expert_margin
        self.supervised_weight = supervised_weight
        super(DQFDLossFunction, self).__init__(scope=scope, **kwargs)

    @rlgraph_api
    def loss(self, q_values_s, actions, rewards, terminals, qt_values_sp, q_values_sp=None, importance_weights=None,
             apply_demo_loss=False):
        loss_per_item = self.loss_per_item(
            q_values_s, actions, rewards, terminals, qt_values_sp, q_values_sp, importance_weights, apply_demo_loss
        )
        total_loss = self.loss_average(loss_per_item)
        return total_loss, loss_per_item

    @rlgraph_api
    def _graph_fn_loss_per_item(self, q_values_s, actions, rewards, terminals,
                                qt_values_sp, q_values_sp=None, importance_weights=None, apply_demo_loss=False):
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
            importance_weights (Optional[SingleDataOp]): If 'self.importance_weights' is True: The batch of weights to
                apply to the losses.
            apply_demo_loss (Optional[SingleDataOp]): If 'apply_demo_loss' is True: The large-margin loss is applied.
                Should be set to True when updating from demo data, False when updating from online data.
        Returns:
            SingleDataOp: The loss values vector (one single value for each batch item).
        """
        if get_backend() == "tf":
            # Make sure the target policy's outputs are treated as constant when calculating gradients.
            qt_values_sp = tf.stop_gradient(qt_values_sp)

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
            qt_sp_ap_values = tf.where(
                condition=terminals, x=tf.zeros_like(qt_sp_ap_values), y=qt_sp_ap_values
            )

            # Q(s,a) -> Use the Q-value of the action actually taken before.
            one_hot = tf.one_hot(indices=actions, depth=self.action_space.num_categories)
            q_s_a_values = tf.reduce_sum(input_tensor=(q_values_s * one_hot), axis=-1)

            # Calculate the TD-delta (target - current estimate).
            td_delta = (rewards + (self.discount ** self.n_step) * qt_sp_ap_values) - q_s_a_values

            # Calculate the demo-loss.
            #  J_E(Q) = max_a([Q(s, a_taken) + l(s, a_expert, a_taken)] - Q(s, a_expert)
            mask = tf.ones_like(tensor=one_hot, dtype=tf.float32)
            action_mask = mask - one_hot
            supervised_loss = tf.reduce_max(input_tensor=q_values_s + action_mask * self.expert_margin, axis=-1)

            # Subtract Q-values of action actually taken.
            supervised_delta = supervised_loss - q_s_a_values
            td_delta = tf.cond(
                pred=apply_demo_loss,
                true_fn=lambda: td_delta + self.supervised_weight * supervised_delta,
                false_fn=lambda: td_delta
            )

            # Reduce over the composite actions, if any.
            if get_rank(td_delta) > 1:
                td_delta = tf.reduce_mean(input_tensor=td_delta, axis=list(range(1, self.ranks_to_reduce + 1)))

            # Apply importance-weights from a prioritized replay to the loss.
            if self.importance_weights:
                return importance_weights * self._apply_huber_loss_if_necessary(td_delta)
            else:
                return self._apply_huber_loss_if_necessary(td_delta)


