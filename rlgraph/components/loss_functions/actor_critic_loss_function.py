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
from rlgraph.spaces import IntBox
from rlgraph.spaces.space_utils import sanity_check_space
from rlgraph.utils.decorators import rlgraph_api, graph_fn

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
    def __init__(self, weight_pg=None, weight_vf=None, weight_entropy=None, **kwargs):
        """
        Args:
            discount (float): The discount factor (gamma) to use.
            gae_lambda (float): Optional GAE discount factor.
            reward_clipping (Optional[str]): One of None, "clamp_one" or "soft_asymmetric". Default: "clamp_one".
            weight_pg (float): The coefficient used for the policy gradient loss term (L[PG]).
            weight_vf (float): The coefficient used for the value function term (L[V]).
            weight_entropy (float): The coefficient used for the entropy regularization term (L[E]).
                In the paper, values between 0.01 and 0.00005 are used via log-uniform search.
        """
        super(ActorCriticLossFunction, self).__init__(scope=kwargs.pop("scope", "actor-critic-loss-func"), **kwargs)

        self.weight_pg = TimeDependentParameter.from_spec(weight_pg if weight_pg is not None else 1.0,
                                                          scope="weight-pg")
        self.weight_vf = TimeDependentParameter.from_spec(
            weight_vf if weight_vf is not None else 0.5, scope="weight-vf"
        )
        self.weight_entropy = TimeDependentParameter.from_spec(
            weight_entropy if weight_entropy is not None else 0.00025, scope="weight-entropy"
        )

        self.add_components(self.weight_pg, self.weight_vf, self.weight_entropy)

    def check_input_spaces(self, input_spaces, action_space=None):
        assert action_space is not None
        self.action_space = action_space
        # Check for IntBox and num_categories.
        sanity_check_space(
            self.action_space, allowed_types=[IntBox], must_have_categories=True
        )

    @rlgraph_api
    def loss(self, log_probs, values, rewards, entropy, time_percentage=None):
        """
        API-method that calculates the total loss (average over per-batch-item loss) from the original input to
        per-item-loss.

        Args: see `self._graph_fn_loss_per_item`.

        Returns:
            SingleDataOp: The tensor specifying the final loss (over the entire batch).
        """
        loss_per_item, vf_loss_per_item = self.loss_per_item(
            log_probs, values, rewards, entropy, time_percentage
        )
        total_loss = self.loss_average(loss_per_item)
        vf_total_loss = self.loss_average(vf_loss_per_item)

        return total_loss, loss_per_item, vf_total_loss, vf_loss_per_item

    @rlgraph_api
    def loss_per_item(self, log_probs, state_values, advantages, entropy, time_percentage=None):
        # Get losses for each action.
        # Baseline loss for V(s) does not depend on actions, only on state.
        vf_loss_per_item = self._graph_fn_state_value_function_loss_per_item(state_values, advantages, time_percentage)
        loss_per_item = self._graph_fn_loss_per_item(log_probs, advantages, entropy, time_percentage)

        # Average across actions.
        loss_per_item = self._graph_fn_average_over_container_keys(loss_per_item)

        return loss_per_item, vf_loss_per_item

    @graph_fn(flatten_ops=True, split_ops=True)
    def _graph_fn_loss_per_item(self, log_probs, advantages, entropy, time_percentage):
        """
        Calculates the loss per batch item (summed over all timesteps) using the formula described above in
        the docstring to this class.

        Args:
            log_probs (DataOp): Log-likelihood of actions.
            advantages (DataOp): The received rewards.
            entropy (DataOp): Policy entropy

        Returns:
            SingleDataOp: The loss values per item in the batch, but summed over all timesteps.
        """
        if get_backend() == "tf":
            # # Let the gae-helper function calculate the pg-advantages.
            advantages = tf.stop_gradient(advantages)

        elif get_backend() == "pytorch":
            advantages = advantages.detach()

        # The policy gradient loss.
        loss = advantages * -log_probs
        loss = self.weight_pg.get(time_percentage) * loss

        # Subtract the entropy bonus from the loss (the larger the entropy the smaller the loss).
        loss -= self.weight_entropy.get(time_percentage) * entropy
        return loss

    @rlgraph_api
    def _graph_fn_state_value_function_loss_per_item(self, state_values, advantages, time_percentage=None):
        """
        Computes the loss for V(s).

        Args:
            state_values (SingleDataOp): Baseline predictions V(s).
            advantages (SingleDataOp): Advantage values.

        Returns:
            SingleDataOp: Baseline loss per item.
        """
        v_targets = None
        if get_backend() == "tf":
            state_values = tf.squeeze(input=state_values, axis=-1)
            v_targets = advantages + state_values
            v_targets = tf.stop_gradient(input=v_targets)
        elif get_backend() == "pytorch":
            state_values = torch.squeeze(state_values, dim=-1)
            v_targets = advantages + state_values
            v_targets = v_targets.detach()

        vf_loss = (v_targets - state_values) ** 2
        return self.weight_vf.get(time_percentage) * vf_loss
