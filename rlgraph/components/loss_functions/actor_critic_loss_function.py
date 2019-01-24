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
from rlgraph.components.helpers import GeneralizedAdvantageEstimation
from rlgraph.components.loss_functions import LossFunction
from rlgraph.spaces import IntBox
from rlgraph.spaces.space_utils import sanity_check_space
from rlgraph.utils.decorators import rlgraph_api

if get_backend() == "tf":
    import tensorflow as tf
elif get_backend() == "pytorch":
    import torch


class ActorCriticLossFunction(LossFunction):
    """
    A basic actor critic policy gradient loss function, including entropy regularization and
    generalized advantage estimation. Suitable for A2C, A3C etc.

    The three terms of the loss function are:
    1) The policy gradient term:
        L[pg] = advantages * nabla log(pi(a|s)).
    2) The value-function baseline term:
        L[V] = 0.5 (vs - V(xs))^2, such that dL[V]/dtheta = (vs - V(xs)) nabla V(xs)
    3) The entropy regularizer term:
        L[E] = - SUM[all actions a] pi(a|s) * log pi(a|s)

    """
    def __init__(self, weight_pg=None, weight_baseline=None, weight_entropy=None, **kwargs):
        """
        Args:
            discount (float): The discount factor (gamma) to use.
            gae_lambda (float): Optional GAE discount factor.
            reward_clipping (Optional[str]): One of None, "clamp_one" or "soft_asymmetric". Default: "clamp_one".
            weight_pg (float): The coefficient used for the policy gradient loss term (L[PG]).
            weight_baseline (float): The coefficient used for the Value-function baseline term (L[V]).
            weight_entropy (float): The coefficient used for the entropy regularization term (L[E]).
                In the paper, values between 0.01 and 0.00005 are used via log-uniform search.
        """
        super(ActorCriticLossFunction, self).__init__(scope=kwargs.pop("scope", "actor-critic-loss-func"), **kwargs)

        self.weight_pg = weight_pg if weight_pg is not None else 1.0
        self.weight_baseline = weight_baseline if weight_baseline is not None else 0.5
        self.weight_entropy = weight_entropy if weight_entropy is not None else 0.00025
        self.action_space = None

    def check_input_spaces(self, input_spaces, action_space=None):
        assert action_space is not None
        self.action_space = action_space
        # Check for IntBox and num_categories.
        sanity_check_space(
            self.action_space, allowed_types=[IntBox], must_have_categories=True
        )

    @rlgraph_api
    def loss(self, logits_actions_pi, action_probs_mu, values, actions, rewards, terminals):
        """
        API-method that calculates the total loss (average over per-batch-item loss) from the original input to
        per-item-loss.

        Args: see `self._graph_fn_loss_per_item`.

        Returns:
            SingleDataOp: The tensor specifying the final loss (over the entire batch).
        """
        loss_per_item, vf_loss_per_item = self.loss_per_item(
            logits_actions_pi, action_probs_mu, values, actions, rewards, terminals
        )
        total_loss = self.loss_average(loss_per_item)
        vf_total_loss = self.loss_average(vf_loss_per_item)

        return total_loss, loss_per_item, vf_total_loss, vf_loss_per_item

    @rlgraph_api
    def _graph_fn_loss_per_item(self, log_probs, entropy, baseline_values, actions,
                                pg_advantages, terminals):
        """
        Calculates the loss per batch item (summed over all timesteps) using the formula described above in
        the docstring to this class.

        Args:
            log_probs (DataOp): Log-likelihood of actions.
            entropy (DataOp): Policy entropy
            baseline_values (DataOp): The state value estimates coming from baseline node of the learner's policy (pi).
            actions (DataOp): The actually taken (already one-hot flattened) actions.
            pg_advantages (DataOp): The received rewards.
            terminals (DataOp): The observed terminal signals.
        Returns:
            SingleDataOp: The loss values per item in the batch, but summed over all timesteps.
        """
        if get_backend() == "tf":
            # # Let the gae-helper function calculate the pg-advantages.
            baseline_values = tf.squeeze(input=baseline_values, axis=-1)

            # Make sure vs and advantage values are treated as constants for the gradient calculation.
            v_targets = pg_advantages + baseline_values
            v_targets = tf.stop_gradient(v_targets)
            pg_advantages = tf.stop_gradient(pg_advantages)

            # The policy gradient loss.
            loss = pg_advantages * -log_probs
            if self.weight_pg != 1.0:
                loss = self.weight_pg * loss

            # The value-function baseline loss.
            baseline_loss = (v_targets - baseline_values) ** 2

            # The entropy regularizer term.
            loss += self.weight_entropy * entropy

            return loss, baseline_loss
        elif get_backend() == "pytorch":
            # # Let the gae-helper function calculate the pg-advantages.
            baseline_values = torch.squeeze(baseline_values, -1)

            # Make sure vs and advantage values are treated as constants for the gradient calculation.
            v_targets = pg_advantages + baseline_values
            v_targets = v_targets.detach()
            pg_advantages = pg_advantages.detach()

            # The policy gradient loss.
            loss = pg_advantages * log_probs
            if self.weight_pg != 1.0:
                loss = self.weight_pg * loss

            # The value-function baseline loss.
            baseline_loss = (v_targets - baseline_values) ** 2

            # The entropy regularizer term.
            loss += self.weight_entropy * entropy
            return loss, baseline_loss
