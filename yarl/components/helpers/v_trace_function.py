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

import tensorflow as tf

from yarl.components import Component
from yarl.spaces.space_utils import sanity_check_space


class VTraceFunction(Component):
    """
    A Helper Component that contains a graph_fn to calculate V-trace values from importance ratios (rhos).
    Based on [1] and coded analogously to: https://github.com/deepmind/scalable_agent

    [1] IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures - Espeholt, Soyer,
        Munos et al. - 2018 (https://arxiv.org/abs/1802.01561)
    """

    def __init__(self, rho_bar=1.0, rho_bar_pg=1.0, c_bar=1.0, clip_pg_rho_threshold=1.0, **kwargs):
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
        super(VTraceFunction, self).__init__(scope=kwargs.pop("scope", "v-trace-function"), **kwargs)

        self.rho_bar = rho_bar
        self.rho_bar_pg = rho_bar_pg
        self.c_bar = c_bar

        # Define our helper API-method, which must not be completed (we don't have variables) and thus its
        # graph_fn can be called anytime from within another graph_fn.
        self.define_api_method("calc_v_trace_values", self._graph_fn_calc_v_trace_values, must_be_complete=False)

    def check_input_spaces(self, input_spaces, action_space):
        in_spaces = input_spaces["calc_v_trace_values"]
        log_rho_space, values_space, bootstrap_value_space, discounts_space, rewards_space, clip_rho_threshold_space, \
            clip_pg_rho_threshold_space = in_spaces

        sanity_check_space(log_rho_space, must_have_batch_rank=True)
        rho_rank = log_rho_space.rank

        # Sanity check our input Spaces for consistency (amongst each other).
        sanity_check_space(values_space, rank=rho_rank, must_have_batch_rank=True)
        sanity_check_space(bootstrap_value_space, rank=rho_rank - 1, must_have_batch_rank=True)
        sanity_check_space(discounts_space, rank=rho_rank, must_have_batch_rank=True)
        sanity_check_space(rewards_space, rank=rho_rank, must_have_batch_rank=True)
        if clip_rho_threshold_space is not None:
            sanity_check_space(clip_rho_threshold_space, rank=0, must_have_batch_rank=False)
        if clip_pg_rho_threshold_space is not None:
            sanity_check_space(clip_pg_rho_threshold_space, rank=0, must_have_batch_rank=False)

    def _graph_fn_calc_v_trace_values(self, log_rhos, discounts, rewards, values, bootstrapped_v):
        """
        Returns the V-trace values calculated from log importance weights (see [1] for details).
        T=time rank
        B=batch rank
        A=action Space

        Args:
            log_rhos (DataOp): The log of the IS (importance sampling) weights:
                log(target_policy(a) / behaviour_policy(a)). Log space is used for numerical stability.
            discounts (DataOp): DataOp (time x batch x values) holding the discounts collected when stepping
                through the environment.
            rewards (DataOp): DataOp (time x batch x values) holding the rewards collected when stepping
                through the environment.
            values (DataOp): A float32 tensor of shape [T, B] with the value function estimates
              wrt. the target policy.
            bootstrapped_v: The last (bootstrapped) to use as a value function estimate after n time steps.

        Returns:
            DataOpTuple:
                v-trace values (vs) in time x batch dimensions used to train the value-function (baseline).
                PG-advantage values in time x batch dimensions used for training via policy gradient with baseline.
        """
        rhos = tf.exp(log_rhos)
        if self.rho_bar is not None:
            rho_t = tf.minimum(self.rho_bar, rhos)
        else:
            rho_t = rhos

        # Apply c-bar clipping to all rhos.
        c_i = tf.minimum(self.c_bar, rhos)
        # Build the vector of Vt from t+1 on until the bootstrapped value after n steps
        # [v1, ..., v_t+1].
        # This is the same vector as `values` except that it will be shifted by 1 timestep to the right.
        values_t_plus_1 = tf.concat([values[1:], tf.expand_dims(bootstrapped_v, 0)], axis=0)
        # Calculate the temporal difference terms (delta-t-V in the paper).
        dt_vs = rho_t * (rewards + discounts * values_t_plus_1 - values)

        # We are doing backwards computation, revert all vectors.
        gamma_c_dtv_at_t = (tf.reverse(discounts, axis=[0]), tf.reverse(c_i, axis=[0]), tf.reverse(dt_vs, axis=[0]))

        # V-trace vs are calculated through a scan from the back to the beginning
        # of the given trajectory.
        def scan_func(acc, gamma_c_dtv_at_t_):
            gamma_t, c_t, dt_v = gamma_c_dtv_at_t_
            return dt_v + gamma_t * c_t * acc

        vs_minus_v_xs = tf.scan(
            fn=scan_func,
            elems=gamma_c_dtv_at_t,
            initializer=tf.zeros_like(bootstrapped_v),
            parallel_iterations=1,
            back_prop=False
        )
        # Reverse the results back to original order.
        vs_minus_v_xs = tf.reverse(vs_minus_v_xs, axis=[0])

        # Add V(x_s) to get v_s.
        vs = tf.add(vs_minus_v_xs, values)

        # Advantage for policy gradient.
        vs_t_plus_1 = tf.concat([vs[1:], tf.expand_dims(bootstrapped_v, 0)], axis=0)
        if self.rho_bar_pg is not None:
            rho_t_pg = tf.minimum(self.rho_bar_pg, rhos)
        else:
            rho_t_pg = rhos
        pg_advantages = rho_t_pg * (rewards + discounts * vs_t_plus_1 - values)

        # Make sure no gradients backpropagated through the returned values.
        return tf.stop_gradient(vs), tf.stop_gradient(pg_advantages)

