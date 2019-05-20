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
from rlgraph.components.common.time_dependent_parameters import TimeDependentParameter
from rlgraph.components.loss_functions import LossFunction
from rlgraph.utils.decorators import rlgraph_api
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
            clip_ratio (Spec[TimeDependentParameter]): How much to clip the likelihood ratio between old and new policy when
                updating.

            value_function_clipping (float): Clipping value for the ValueFunction component.
                If None, no clipping is applied.

            weight_entropy (Optional[Spec[TimeDependentParameter]]): The weight with which to multiply the entropy and subtract
                from the loss.
        """
        super(PPOLossFunction, self).__init__(scope=scope, **kwargs)

        self.clip_ratio = TimeDependentParameter.from_spec(clip_ratio, scope="clip-ratio")
        self.weight_entropy = TimeDependentParameter.from_spec(
            weight_entropy if weight_entropy is not None else 0.00025, scope="weight-entropy"
        )
        self.value_function_clipping = value_function_clipping
        self.ranks_to_reduce = None
        self.expand_entropy = None  # Whether to expand the entropy input by one rank.

        self.add_components(self.clip_ratio, self.weight_entropy)

    def check_input_spaces(self, input_spaces, action_space=None):
        """
        Do some sanity checking on the incoming Spaces:
        """
        assert action_space is not None
        self.action_space = action_space.with_batch_rank()
        #self.flat_action_space = action_space.flatten()
        #sanity_check_space(self.action_space, must_have_batch_rank=True)
        self.ranks_to_reduce = len(self.action_space.get_shape(with_batch_rank=True)) - 1
        self.expand_entropy = len(input_spaces["entropy"].get_shape(with_batch_rank=True)) == 1

    @rlgraph_api
    def loss(self, log_probs, prev_log_probs, state_values, prev_state_values, advantages, entropy, time_percentage):
        """
        API-method that calculates the total loss (average over per-batch-item loss) from the original input to
        per-item-loss.

        Args: see `self._graph_fn_loss_per_item`.

        Returns:
            Total loss, loss per item, total value-function loss, value-function loss per item.
        """
        loss_per_item, vf_loss_per_item = self.loss_per_item(
            log_probs, prev_log_probs, state_values, prev_state_values, advantages, entropy, time_percentage
        )
        total_loss = self.loss_average(loss_per_item)
        total_vf_loss = self.loss_average(vf_loss_per_item)

        return total_loss, loss_per_item, total_vf_loss, vf_loss_per_item

    @rlgraph_api
    def loss_per_item(self, log_probs, prev_log_probs, state_values, prev_state_values, advantages,
                      entropy, time_percentage):
        # Get losses for each action.
        # Baseline loss for V(s) does not depend on actions, only on state.
        pg_loss_per_item = self.pg_loss_per_item(log_probs, prev_log_probs, advantages, entropy, time_percentage)
        vf_loss_per_item = self.value_function_loss_per_item(state_values, prev_state_values, advantages)

        # Average PG-loss across action components.
        pg_loss_per_item = self._graph_fn_average_over_container_keys(pg_loss_per_item)

        return pg_loss_per_item, vf_loss_per_item

    @rlgraph_api(flatten_ops=True, split_ops=True)
    def _graph_fn_pg_loss_per_item(self, log_probs, prev_log_probs, advantages, entropy, time_percentage):
        """
        Args:
            log_probs (SingleDataOp): Log-likelihoods of actions under policy.
            prev_log_probs (SingleDataOp) Log-likelihoods of actions under policy before this update step.
            advantages (SingleDataOp): The batch of post-processed generalized advantage estimations (GAEs).
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
                ratio = tf.squeeze(ratio, axis=-1)
                entropy = tf.squeeze(entropy, axis=-1)

            clipped_advantages = tf.where(
                condition=advantages > 0,
                x=(1 + self.clip_ratio.get(time_percentage)) * advantages,
                y=(1 - self.clip_ratio.get(time_percentage)) * advantages
            )
            #clipped_advantages = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            loss = -tf.minimum(x=ratio * advantages, y=clipped_advantages)

            # Subtract the entropy bonus from the loss (the larger the entropy the smaller the loss).
            loss -= self.weight_entropy.get(time_percentage) * entropy

            # Reduce over the composite actions, if any.
            #if self.ranks_to_reduce > 0:
            #    loss = tf.reduce_mean(loss, axis=list(range(1, self.ranks_to_reduce + 1)))

            return loss

        elif get_backend() == "pytorch":
            # Likelihood ratio and clipped objective.
            ratio = torch.exp(log_probs - prev_log_probs)

            # Make sure the pg_advantages vector (batch) is broadcast correctly.
            for _ in range(get_rank(ratio) - 1):
                ratio = torch.squeeze(ratio, dim=-1)
                entropy = torch.squeeze(entropy, dim=-1)

            clipped_advantages = torch.where(
                advantages > 0,
                (1 + self.clip_ratio.get(time_percentage)) * advantages,
                (1 - self.clip_ratio.get(time_percentage)) * advantages
            )
            #clipped_advantages = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            loss = -torch.min(ratio * advantages, clipped_advantages)

            # Subtract the entropy bonus from the loss (the larger the entropy the smaller the loss).
            loss -= self.weight_entropy.get(time_percentage) * entropy

            # Reduce over the composite actions, if any.
            if self.ranks_to_reduce > 0:
                loss = torch.mean(loss, tuple(range(1, self.ranks_to_reduce + 1)), keepdim=False)

            return loss

    @rlgraph_api
    def _graph_fn_value_function_loss_per_item(self, state_values, prev_state_values, advantages):
        """
        Computes the loss for V(s).

        Args:
            state_values (SingleDataOp): State value predictions V(s).
            prev_state_values (SingleDataOp): Previous state value predictions V(s) (before the update).
            advantages (SingleDataOp): GAE (advantage) values.

        Returns:
            SingleDataOp: Value function loss per item.
        """
        if get_backend() == "tf":
            state_values = tf.squeeze(input=state_values, axis=-1)
            prev_state_values = tf.squeeze(input=prev_state_values, axis=-1)
            v_targets = advantages + prev_state_values
            v_targets = tf.stop_gradient(input=v_targets)
            vf_loss = (state_values - v_targets) ** 2
            if self.value_function_clipping:
                vf_clipped = prev_state_values + tf.clip_by_value(
                    state_values - prev_state_values, -self.value_function_clipping, self.value_function_clipping
                )
                clipped_loss = (vf_clipped - v_targets) ** 2
                return tf.maximum(vf_loss, clipped_loss)
            else:
                return vf_loss

        elif get_backend() == "pytorch":
            state_values = torch.squeeze(state_values, dim=-1)
            prev_state_values = torch.squeeze(input=prev_state_values, dim=-1)
            v_targets = advantages + prev_state_values
            v_targets = v_targets.detach()
            vf_loss = (state_values - v_targets) ** 2
            if self.value_function_clipping:
                vf_clipped = prev_state_values + torch.clamp(
                    state_values - prev_state_values, -self.value_function_clipping, self.value_function_clipping
                )
                clipped_loss = (vf_clipped - v_targets) ** 2
                return torch.max(vf_loss, clipped_loss)
            else:
                return vf_loss
