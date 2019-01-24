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

import numpy as np

from rlgraph import get_backend
from rlgraph.components import Component
from rlgraph.utils.decorators import rlgraph_api
from rlgraph.utils.numpy import softmax

if get_backend() == "tf":
    import tensorflow as tf


class VTraceFunction(Component):
    """
    A Helper Component that contains a graph_fn to calculate V-trace values from importance ratios (rhos).
    Based on [1] and coded analogously to: https://github.com/deepmind/scalable_agent

    [1] IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures - Espeholt, Soyer,
        Munos et al. - 2018 (https://arxiv.org/abs/1802.01561)
    """

    def __init__(self, rho_bar=1.0, rho_bar_pg=1.0, c_bar=1.0, device="/device:CPU:0",
                 scope="v-trace-function", **kwargs):
        """
        Args:
            rho_bar (float): The maximum values of the IS-weights for the temporal differences of V.
                Use None for not applying any clipping.
            rho_bar_pg (float): The maximum values of the IS-weights for the policy-gradient loss:
                \rho_s \delta log \pi(a|x) (r + \gamma v_{s+1} - V(x_s))
                Use None for not applying any clipping.
            c_bar (float): The maximum values of the IS-weights for the time trace.
                Use None for not applying any clipping.
        """
        super(VTraceFunction, self).__init__(device=device, space_agnostic=True, scope=scope, **kwargs)

        self.rho_bar = rho_bar
        self.rho_bar_pg = rho_bar_pg
        self.c_bar = c_bar

    def check_input_spaces(self, input_spaces, action_space=None):
        pass
        # TODO: Complete all input arg checks.
        #log_is_weight_space = input_spaces["log_is_weights"]
        #discounts_space, rewards_space, values_space, bootstrap_value_space = \
        #    input_spaces["discounts"], input_spaces["rewards"], \
        #    input_spaces["values"], input_spaces["bootstrapped_values"]

        #sanity_check_space(log_is_weight_space, must_have_batch_rank=True)
        #log_is_weight_rank = log_is_weight_space.rank

        # Sanity check our input Spaces for consistency (amongst each other).
        #sanity_check_space(values_space, rank=log_is_weight_rank, must_have_batch_rank=True, must_have_time_rank=True)
        #sanity_check_space(bootstrap_value_space, must_have_batch_rank=True, must_have_time_rank=True)
        #sanity_check_space(discounts_space, rank=log_is_weight_rank,
        #                   must_have_batch_rank=True, must_have_time_rank=True)
        #sanity_check_space(rewards_space, rank=log_is_weight_rank, must_have_batch_rank=True, must_have_time_rank=True)

    @rlgraph_api(must_be_complete=False)
    def _graph_fn_calc_v_trace_values(self, logits_actions_pi, log_probs_actions_mu, actions, actions_flat,
                                      discounts, rewards,
                                      values, bootstrapped_values):
        """
        Returns the V-trace values calculated from log importance weights (see [1] for details).
        Calculation:
        vs = V(xs) + SUM[t=s to s+N-1]( gamma^t-s * ( PROD[i=s to t-1](ci) ) * dt_V )
        with:
            dt_V = rho_t * (rt + gamma V(xt+1) - V(xt))
            rho_t and ci being the clipped IS weights

        Args:
            logits_actions_pi (SingleDataOp): The raw logits output of the pi-network (one logit per discrete action).
            log_probs_actions_mu (SingleDataOp): The log-probs of the mu-network (one log-prob per discrete action).
            actions (SingleDataOp): The (int encoded) actually taken actions.
            actions_flat (SingleDataOp): The one-hot converted actually taken actions.
            discounts (SingleDataOp): DataOp (time x batch x values) holding the discounts collected when stepping
                through the environment (for the timesteps s=t to s=t+N-1).
            rewards (SingleDataOp): DataOp (time x batch x values) holding the rewards collected when stepping
                through the environment (for the timesteps s=t to s=t+N-1).
            values (SingleDataOp): DataOp (time x batch x values) holding the the value function estimates
                wrt. the learner's policy (pi) (for the timesteps s=t to s=t+N-1).
            bootstrapped_values (SingleDataOp): DataOp (time(1) x batch x values) holding the last (bootstrapped)
                value estimate to use as a value function estimate after n time steps (V(xs) for s=t+N).

        Returns:
            tuple:
                - v-trace values (vs) in time x batch dimensions used to train the value-function (baseline).
                - PG-advantage values in time x batch dimensions used for training via policy gradient with baseline.
        """
        # Simplified (not performance optimized!) numpy implementation of v-trace for testing purposes.
        if get_backend() == "python" or self.backend == "python":
            probs_actions_pi = softmax(logits_actions_pi, axis=-1)
            log_probs_actions_pi = np.log(probs_actions_pi)

            log_is_weights = log_probs_actions_pi - log_probs_actions_mu  # log(a/b) = log(a) - log(b)
            log_is_weights_actions_taken = np.sum(log_is_weights * actions_flat, axis=-1, keepdims=True)
            is_weights = np.exp(log_is_weights_actions_taken)

            # rho_t = min(rho_bar, is_weights) = [1.0, 1.0], [0.67032005, 1.0], [1.0, 0.36787944]
            if self.rho_bar is not None:
                rho_t = np.minimum(self.rho_bar, is_weights)
            else:
                rho_t = is_weights

            # Same for rho-PG (policy gradients).
            if self.rho_bar_pg is not None:
                rho_t_pg = np.minimum(self.rho_bar_pg, is_weights)
            else:
                rho_t_pg = is_weights

            # Calculate ci terms for all timesteps:
            # ci = min(c_bar, is_weights) = [1.0, 1.0], [0.67032005, 1.0], [1.0, 0.36787944]
            if self.c_bar is not None:
                c_i = np.minimum(self.c_bar, is_weights)
            else:
                c_i = is_weights

            # Values t+1 -> shift by one time step.
            values_t_plus_1 = np.concatenate((values[1:], bootstrapped_values), axis=0)
            deltas = rho_t * (rewards + discounts * values_t_plus_1 - values)

            # Reverse everything for recursive v_s calculation.
            discounts_reversed = discounts[::-1]
            c_i_reversed = c_i[::-1]
            deltas_reversed = deltas[::-1]

            vs_minus_v_xs = [np.zeros_like(np.squeeze(bootstrapped_values, axis=0))]

            # Do the recursive calculations.
            for d, c, delta in zip(discounts_reversed, c_i_reversed, deltas_reversed):
                vs_minus_v_xs.append(delta + d * c * vs_minus_v_xs[-1])

            # Convert into numpy array and revert back.
            vs_minus_v_xs = np.array(vs_minus_v_xs[::-1])[:-1]

            # Add V(x_s) to get v_s.
            vs = vs_minus_v_xs + values

            # Advantage for policy gradient.
            vs_t_plus_1 = np.concatenate([vs[1:], bootstrapped_values], axis=0)
            pg_advantages = (rho_t_pg * (rewards + discounts * vs_t_plus_1 - values))

            return vs, pg_advantages

        elif get_backend() == "tf":
            # Calculate the log IS-weight values via: logIS = log(pi(a|s)) - log(mu(a|s)).
            # Use the action_probs_pi values only of the actions actually taken.
            log_probs_actions_taken_pi = tf.expand_dims(-tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits_actions_pi, labels=actions
            ), axis=-1)
            log_probs_actions_taken_mu = tf.reduce_sum(
                input_tensor=log_probs_actions_mu * actions_flat, axis=-1, keepdims=True, name="log-probs-actions-taken-mu"
            )
            log_is_weights = log_probs_actions_taken_pi - log_probs_actions_taken_mu

            is_weights = tf.exp(x=log_is_weights, name="is-weights-from-logs")

            # Apply rho-bar (also for PG) and c-bar clipping to all IS-weights.
            if self.rho_bar is not None:
                rho_t = tf.minimum(x=self.rho_bar, y=is_weights, name="clip-rho-bar")
            else:
                rho_t = is_weights

            if self.rho_bar_pg is not None:
                rho_t_pg = tf.minimum(x=self.rho_bar_pg, y=is_weights, name="clip-rho-bar-pg")
            else:
                rho_t_pg = is_weights

            if self.c_bar is not None:
                c_i = tf.minimum(x=self.c_bar, y=is_weights, name="clip-c-bar")
            else:
                c_i = is_weights

            # This is the same vector as `values` except that it will be shifted by 1 timestep to the right and
            # include - as the last item - the bootstrapped V value at s=t+N.
            values_t_plus_1 = tf.concat(values=[values[1:], bootstrapped_values], axis=0, name="values-t-plus-1")
            # Calculate the temporal difference terms (delta-t-V in the paper) for each s=t to s=t+N-1.
            dt_vs = rho_t * (rewards + discounts * values_t_plus_1 - values)

            # V-trace values can be calculated recursively (starting from the end of a trajectory) via:
            #    vs = V(xs) + dsV + gamma * cs * (vs+1 - V(s+1))
            # => (vs - V(xs)) = dsV + gamma * cs * (vs+1 - V(s+1))
            # We will thus calculate all terms: [vs - V(xs)] for all timesteps first, then add V(xs) again to get the
            # v-traces.
            elements = (
                tf.reverse(tensor=discounts, axis=[0], name="revert-discounts"),
                tf.reverse(tensor=c_i, axis=[0], name="revert-c-i"),
                tf.reverse(tensor=dt_vs, axis=[0], name="revert-dt-vs")
            )

            def scan_func(vs_minus_v_xs_, elements_):
                gamma_t, c_t, dt_v = elements_
                return dt_v + gamma_t * c_t * vs_minus_v_xs_

            vs_minus_v_xs = tf.scan(
                fn=scan_func,
                elems=elements,
                initializer=tf.zeros_like(tensor=tf.squeeze(bootstrapped_values, axis=0)),
                parallel_iterations=1,
                back_prop=False,
                name="v-trace-scan"
            )
            # Reverse the results back to original order.
            vs_minus_v_xs = tf.reverse(tensor=vs_minus_v_xs, axis=[0], name="revert-vs-minus-v-xs")

            # Add V(xs) to get vs.
            vs = tf.add(x=vs_minus_v_xs, y=values)

            # Calculate the advantage values (for policy gradient loss term) according to:
            # A = Q - V with Q based on vs (v-trace) values: qs = rs + gamma * vs and V being the
            # approximate value function output.
            vs_t_plus_1 = tf.concat(values=[vs[1:], bootstrapped_values], axis=0)
            pg_advantages = rho_t_pg * (rewards + discounts * vs_t_plus_1 - values)

            # Return v-traces and policy gradient advantage values based on: A=r+gamma*v-trace(s+1) - V(s).
            # With `r+gamma*v-trace(s+1)` also called `qs` in the paper.
            return tf.stop_gradient(vs), tf.stop_gradient(pg_advantages)
