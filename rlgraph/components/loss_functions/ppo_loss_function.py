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
from rlgraph.components import rlgraph_api
from rlgraph.components.helpers import GeneralizedAdvantageEstimation
from rlgraph.components.loss_functions import LossFunction
from rlgraph.utils.decorators import graph_fn

if get_backend() == "tf":
    import tensorflow as tf


class PPOLossFunction(LossFunction):
    """
    Loss function for proximal policy optimization:

    https://arxiv.org/abs/1707.06347
    """
    def __init__(self, discount=0.99,  gae_lambda=1.0, clip_ratio=0.2, scope="ppo-loss-function", **kwargs):
        """
        Args:
            discount (float): The discount factor (gamma) to use.
            gae_lambda (float): Optional GAE discount factor.
            clip_ratio (float): How much to clip the likelihood ratio between old and new policy when updating.
            **kwargs:
        """
        self.clip_ratio = clip_ratio
        super(PPOLossFunction, self).__init__(scope=scope, **kwargs)

        self.action_space = None
        self.gae_function = GeneralizedAdvantageEstimation(gae_lambda=gae_lambda, discount=discount)

    @rlgraph_api
    def loss(self, log_probs, baseline_values, actions, rewards, terminals, prev_log_probs):
        """
        API-method that calculates the total loss (average over per-batch-item loss) from the original input to
        per-item-loss.

        Args: see `self._graph_fn_loss_per_item`.

        Returns:
            Total loss, loss per item, total baseline loss, baseline loss per item.
        """
        loss_per_item, baseline_loss_per_item = self.loss_per_item(
            log_probs, baseline_values, actions, rewards, terminals, prev_log_probs
        )
        total_loss = self.loss_average(loss_per_item)
        total_baseline_loss = self.loss_average(baseline_loss_per_item)

        return total_loss, loss_per_item, total_baseline_loss, baseline_loss_per_item

    @graph_fn
    def _graph_fn_loss_per_item(self, log_probs, baseline_values, actions, rewards, terminals, prev_log_probs):
        """
        Args:
            actions (SingleDataOp): The batch of actions that were actually taken in states s (from a memory).
            rewards (SingleDataOp): The batch of rewards that we received after having taken a in s (from a memory).
            terminals (SingleDataOp): The batch of terminal signals that we received after having taken a in s
                (from a memory).
            prev_log_likelihood (SingleDataOp): Log likelihood to compare to when computing likelihood ratios.
        Returns:
            SingleDataOp: The loss values vector (one single value for each batch item).
        """
        if get_backend() == "tf":
            # Compute advantages.
            baseline_values = tf.squeeze(input=baseline_values, axis=-1)
            pg_advantages = self.gae_function.calc_gae_values(baseline_values, rewards, terminals)
            v_targets = pg_advantages + baseline_values
            v_targets = tf.stop_gradient(v_targets)

            # Likelihood ratio and clipped objective.
            ratio = tf.exp(log_probs - prev_log_probs)
            clipped_advantages = tf.where(
                pg_advantages > 0,
                (1 + self.clip_ratio) * pg_advantages,
                (1 - self.clip_ratio) * pg_advantages
            )

            loss = -tf.reduce_mean(input_tensor=tf.minimum(x=ratio * pg_advantages, y=clipped_advantages))
            baseline_loss = tf.reduce_mean((v_targets - baseline_values) ** 2)

            return loss, baseline_loss
