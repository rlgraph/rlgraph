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
from rlgraph.components.helpers.v_trace_function import VTraceFunction
from rlgraph.components.loss_functions import LossFunction
from rlgraph.spaces import IntBox
from rlgraph.spaces.space_utils import sanity_check_space

if get_backend() == "tf":
    import tensorflow as tf


class IMPALALossFunction(LossFunction):
    """
    The IMPALA loss function based on v-trace off-policy policy gradient corrections, described in detail in [1].

    The three terms of the loss function are:
    1) The policy gradient term:
        L[pg] = (rho_pg * advantages) * nabla log(pi(a|s)), where (rho_pg * advantages)=pg_advantages in code below.
    2) The value-function baseline term:
        L[V] = 0.5 (vs - V(xs))^2, such that dL[V]/dtheta = (vs - V(xs)) nabla V(xs)
    3) The entropy regularizer term:
        L[E] = - SUM[all actions a] pi(a|s) * log pi(a|s)

    [1] IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures - Espeholt, Soyer,
        Munos et al. - 2018 (https://arxiv.org/abs/1802.01561)
    """
    def __init__(self, discount=0.99, reward_clipping="clamp_one",
                 weight_pg=None, weight_baseline=None, weight_entropy=None, **kwargs):
        """
        Args:
            discount (float): The discount factor (gamma) to use.
            reward_clipping (Optional[str]): One of None, "clamp_one" or "soft_asymmetric". Default: "clamp_one".
            weight_pg (float): The coefficient used for the policy gradient loss term (L[PG]).
            weight_baseline (float): The coefficient used for the Value-function baseline term (L[V]).
            weight_entropy (float): The coefficient used for the entropy regularization term (L[E]).
                In the paper, values between 0.01 and 0.00005 are used via log-uniform search.
        """
        # graph_fn_num_outputs=dict(_graph_fn_loss_per_item=2) <- debug
        super(IMPALALossFunction, self).__init__(scope=kwargs.pop("scope", "impala-loss-func"), **kwargs)

        self.discount = discount
        self.v_trace_function = VTraceFunction()

        self.reward_clipping = reward_clipping

        self.weight_pg = weight_pg if weight_pg is not None else 1.0
        self.weight_baseline = weight_baseline if weight_baseline is not None else 0.5
        self.weight_entropy = weight_entropy if weight_entropy is not None else 0.00025

        self.action_space = None

        self.add_components(self.v_trace_function)

    def check_input_spaces(self, input_spaces, action_space=None):
        assert action_space is not None
        self.action_space = action_space
        # Check for IntBox and num_categories.
        sanity_check_space(
            self.action_space, allowed_types=[IntBox], must_have_categories=True
        )

    def loss(self, logits_actions_pi, action_probs_mu, values, actions, rewards, terminals):
        """
        API-method that calculates the total loss (average over per-batch-item loss) from the original input to
        per-item-loss.

        Args: see `self._graph_fn_loss_per_item`.

        Returns:
            SingleDataOp: The tensor specifying the final loss (over the entire batch).
        """
        #fake_step_op,
        loss_per_item = self._graph_fn_loss_per_item(
            logits_actions_pi, action_probs_mu, values, actions, rewards, terminals
        )
        total_loss = self._graph_fn_loss_average(loss_per_item)
        # TODO: REMOVE no_op again. Only for IMPALA testing w/o update step.
        # fake_step_op,
        return total_loss, loss_per_item

    def _graph_fn_loss_per_item(self, logits_actions_pi, action_probs_mu, values, actions,
                                rewards, terminals):  #, bootstrapped_values):
        """
        Calculates the loss per batch item (summed over all timesteps) using the formula described above in
        the docstring to this class.

        Args:
            logits_actions_pi (DataOp): The logits for all possible actions coming from the learner's
                policy (pi). Dimensions are: (time+1) x batch x action-space+categories.
                +1 b/c last-next-state (aka "bootstrapped" value).
            action_probs_mu (DataOp): The probabilities for all actions coming from the
                actor's policies (mu). Dimensions are: (time+1) x batch x action-space+categories.
            values (DataOp): The state value estimates coming from baseline node of the learner's policy (pi).
                Dimensions are: (time+1) x batch. +1 b/c last-next-state (aka "bootstrapped" value).
            actions (DataOp): The actually taken (already one-hot flattened) actions.
                Dimensions are: (time+1) x batch x N (N=number of discrete actions).
            rewards (DataOp): The received rewards. Dimensions are: (time+1) x batch.
            terminals (DataOp): The observed terminal signals. Dimensions are: (time+1) x batch.

        Returns:
            SingleDataOp: The loss values per item in the batch, but summed over all timesteps.
        """
        if get_backend() == "tf":
            values, bootstrapped_values = values[:-1], values[-1:]

            #return tf.no_op(), tf.ones_like(tf.squeeze(bootstrapped_values, axis=0))

            logits_actions_pi = logits_actions_pi[:-1]
            # Ignore very first actions/rewards (these are the previous ones only used as part of the state input
            # for the network)
            actions_flat = actions[1:]
            actions = tf.reduce_sum(
                tf.cast(actions_flat * tf.range(self.action_space.num_categories, dtype=tf.float32), dtype=tf.int32),
                axis=-1
            )
            rewards = rewards[1:]
            terminals = terminals[1:]
            action_probs_mu = action_probs_mu[1:]

            # Discounts are simply 0.0, if there is a terminal, otherwise: `discount`.
            discounts = tf.expand_dims(tf.to_float(~terminals) * self.discount, axis=-1, name="discounts")
            # `clamp_one`: Clamp rewards between -1.0 and 1.0.
            if self.reward_clipping == "clamp_one":
                rewards = tf.clip_by_value(rewards, -1, 1, name="reward-clipping")
            # `soft_asymmetric`: Negative rewards are less negative than positive rewards are positive.
            elif self.reward_clipping == "soft_asymmetric":
                squeezed = tf.tanh(rewards / 5.0)
                rewards = tf.where(rewards < 0.0, 0.3 * squeezed, squeezed) * 5.0

            # Let the v-trace  helper function calculate the v-trace values (vs) and the pg-advantages
            # (already multiplied by rho_t_pg): A = rho_t_pg * (rt + gamma*vt - V(t)).
            # Both vs and pg_advantages will block the gradient as they should be treated as constants by the gradient
            # calculator of this loss func.
            vs, pg_advantages = self.v_trace_function.calc_v_trace_values(
                logits_actions_pi, tf.log(action_probs_mu), actions, actions_flat, discounts, rewards, values,
                bootstrapped_values
            )

            cross_entropy = tf.expand_dims(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=actions, logits=logits_actions_pi
            ), axis=-1)

            #vs = tf.ones_like(values)
            #pg_advantages = tf.ones_like(log_probs_actions_taken_pi)

            # Make sure vs and advantage values are treated as constants for the gradient calculation.
            #vs = tf.stop_gradient(vs)
            pg_advantages = tf.stop_gradient(pg_advantages)

            # The policy gradient loss.
            loss_pg = pg_advantages * cross_entropy
            loss = tf.reduce_sum(loss_pg, axis=0)  # reduce over the time-rank
            if self.weight_pg != 1.0:
                loss = self.weight_pg * loss

            # The value-function baseline loss.
            loss_baseline = 0.5 * tf.square(x=tf.subtract(vs, values))
            loss_baseline = tf.reduce_sum(loss_baseline, axis=0)  # reduce over the time-rank
            loss += self.weight_baseline * loss_baseline

            # The entropy regularizer term.
            policy = tf.nn.softmax(logits=logits_actions_pi)
            log_policy = tf.nn.log_softmax(logits=logits_actions_pi)
            loss_entropy = tf.reduce_sum(-policy * log_policy, axis=-1)
            loss_entropy = -tf.reduce_sum(loss_entropy, axis=0)  # reduce over the time-rank
            loss += self.weight_entropy * loss_entropy

            return loss
