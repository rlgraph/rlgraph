# Copyright 2018/2019 The RLgraph authors. All Rights Reserved.
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
from rlgraph.components.loss_functions import LossFunction
from rlgraph.utils.decorators import rlgraph_api, graph_fn
from rlgraph.utils.util import get_rank

if get_backend() == "tf":
    import tensorflow as tf
elif get_backend() == "pytorch":
    import torch


class PPOLossFunction(LossFunction):
    """
    Loss function for proximal policy optimization:

    https://arxiv.org/abs/1707.06347
    """
    def __init__(self, clip_ratio=0.2, value_function_clipping=None, weight_entropy=None,
                 scope="ppo-loss-function", **kwargs):
        """
        Args:
            clip_ratio (float): How much to clip the likelihood ratio between old and new policy when updating.
            value_function_clipping (float): Clipping value for the baseline object. If None, no clipping is applied.
            **kwargs:
        """
        self.clip_ratio = clip_ratio
        self.weight_entropy = weight_entropy if weight_entropy is not None else 0.00025
        self.value_function_clipping = value_function_clipping
        self.ranks_to_reduce = None

        super(PPOLossFunction, self).__init__(scope=scope, **kwargs)

    def check_input_spaces(self, input_spaces, action_space=None):
        """
        Do some sanity checking on the incoming Spaces:
        """
        assert action_space is not None
        self.action_space = action_space.with_batch_rank()
        #self.flat_action_space = action_space.flatten()
        #sanity_check_space(self.action_space, must_have_batch_rank=True)
        self.ranks_to_reduce = len(self.action_space.get_shape(with_batch_rank=True)) - 1

    @rlgraph_api
    def loss(self, log_probs, prev_log_probs, baseline_values, prev_baseline_values, advantages, entropy):
        """
        API-method that calculates the total loss (average over per-batch-item loss) from the original input to
        per-item-loss.

        Args: see `self._graph_fn_loss_per_item`.

        Returns:
            Total loss, loss per item, total baseline loss, baseline loss per item.
        """
        loss_per_item, baseline_loss_per_item = self.loss_per_item(
            log_probs, prev_log_probs, baseline_values, prev_baseline_values, advantages, entropy
        )
        total_loss = self.loss_average(loss_per_item)
        total_baseline_loss = self.loss_average(baseline_loss_per_item)

        return total_loss, loss_per_item, total_baseline_loss, baseline_loss_per_item

    @rlgraph_api
    def loss_per_item(self, log_probs, prev_log_probs, baseline_values, prev_baseline_values, advantages, entropy):
        # Get losses for each action.
        # Baseline loss for V(s) does not depend on actions, only on state.
        baseline_loss_per_item = self._graph_fn_baseline_loss_per_item(baseline_values, prev_baseline_values, advantages)
        loss_per_item = self._graph_fn_loss_per_item(log_probs, prev_log_probs, advantages, entropy)

        # Average across action components.
        loss_per_item = self._graph_fn_average_over_container_keys(loss_per_item)

        return loss_per_item, baseline_loss_per_item

    @graph_fn(flatten_ops=True, split_ops=True)
    def _graph_fn_loss_per_item(self, log_probs, prev_log_probs, advantages, entropy):
        """
        Args:
            log_probs (SingleDataOp): Log-likelihoods of actions under policy.
            advantages (SingleDataOp): The batch of post-processed advantages.
            entropy (SingleDataOp): Policy entropy.

        Returns:
            SingleDataOp: The loss values vector (one single value for each batch item).
        """
        if get_backend() == "tf":
            # N.b.: Many implementations do the following:
            # Sample action -> return policy log probs with action -> feed both back in from memory/via placeholders.
            # This creates the same effect as just stopping the gradients on the log-probs.
            # Saving them would however remove necessity for an extra forward pass.
            # Likelihood ratio and clipped objective.
            ratio = tf.exp(x=log_probs - prev_log_probs)

            # Make sure the pg_advantages vector (batch) is broadcast correctly.
            for _ in range(get_rank(ratio) - 1):
                advantages = tf.expand_dims(advantages, axis=1)

            clipped_advantages = tf.where(
                condition=advantages > 0,
                x=(1 + self.clip_ratio) * advantages,
                y=(1 - self.clip_ratio) * advantages
            )
            loss = -tf.minimum(x=ratio * advantages, y=clipped_advantages)
            loss += self.weight_entropy * entropy

            # Reduce over the composite actions, if any.
            if get_rank(loss) > 1:
                loss = tf.reduce_mean(loss, axis=list(range(1, self.ranks_to_reduce + 1)))

            return tf.squeeze(loss)

        elif get_backend() == "pytorch":
            # Likelihood ratio and clipped objective.
            ratio = torch.exp(log_probs - prev_log_probs)

            # Make sure the pg_advantages vector (batch) is broadcast correctly.
            for _ in range(get_rank(ratio) - 1):
                advantages = torch.unsqueeze(advantages, dim=1)

            clipped_advantages = torch.where(
                advantages > 0,
                (1 + self.clip_ratio) * advantages,
                (1 - self.clip_ratio) * advantages
            )

            loss = -torch.min(ratio * advantages, clipped_advantages)
            loss += self.weight_entropy * entropy

            # Reduce over the composite actions, if any.
            if get_rank(loss) > 1:
                loss = torch.mean(loss, tuple(range(1, self.ranks_to_reduce + 1)), keepdim=False)

            return torch.squeeze(loss, dim=-1)

    @rlgraph_api
    def _graph_fn_baseline_loss_per_item(self, baseline_values, prev_baseline_values, advantages):
        """
        Computes the loss for V(s).

        Args:
            baseline_values (SingleDataOp): Baseline predictions V(s).
            prev_baseline_values (SingleDataOp): Previous baseline predictions V(s).
            advantages (SingleDataOp): Advantage values.

        Returns:
            SingleDataOp: Baseline loss per item.
        """
        if get_backend() == "tf":
            baseline_values = tf.squeeze(input=baseline_values, axis=-1)
            v_targets = advantages + baseline_values
            v_targets = tf.stop_gradient(input=v_targets)
            baseline_loss = (v_targets - baseline_values) ** 2
            if self.value_function_clipping is not None:
                prev_baseline_values = tf.squeeze(input=prev_baseline_values, axis=-1)
                vf_clipped = prev_baseline_values + tf.clip_by_value(
                    baseline_values - prev_baseline_values, -self.value_function_clipping, self.value_function_clipping
                )
                clipped_loss = (v_targets - vf_clipped) ** 2
                return tf.squeeze(tf.maximum(baseline_loss, clipped_loss))
            else:
                return tf.squeeze(baseline_loss)

        elif get_backend() == "pytorch":
            baseline_values = torch.squeeze(baseline_values, dim=-1)

            v_targets = advantages + baseline_values
            v_targets = v_targets.detach()
            baseline_loss = (v_targets - baseline_values) ** 2
            if self.value_function_clipping is not None:
                prev_baseline_values = torch.squeeze(input=prev_baseline_values, dim=-1)
                vf_clipped = prev_baseline_values + torch.clamp(
                    baseline_values - prev_baseline_values, -self.value_function_clipping, self.value_function_clipping
                )
                clipped_loss = (v_targets - vf_clipped) ** 2
                return torch.squeeze(torch.max(baseline_loss, clipped_loss), dim=-1)
            else:
                return torch.squeeze(baseline_loss, dim=-1)
