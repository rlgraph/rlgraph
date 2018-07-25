# Copyright 2018 The RLgraph authors, All Rights Reserved.
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

if get_backend() == "tf":
    import tensorflow as tf


class IMPALALossFunction(LossFunction):
    """
    The IMPALA loss function based on v-trace off-policy policy gradient corrections, described in detail in [1].
    The three terms of the loss function are:
    1) The policy gradient term:
        L[pg] = rho_pg * nabla log(pi(a|s)) * Advantage
    2) The value-function baseline term:
        L[V] = 0.5 (vs - V(xs))^2 -> so that dL/dtheta = (vs - V(xs)) nabla V(xs)
    3) The entropy regulariser term:
        L[E] = - SUM[all actions a] pi(a|s) * log pi(a|s)

    [1] IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures - Espeholt, Soyer,
        Munos et al. - 2018 (https://arxiv.org/abs/1802.01561)
    """
    def __init__(self, v_trace_spec=None, weight_pg=1.0, weight_baseline=1.0, weight_entropy=0.001, **kwargs):
        """
        Args:
            v_trace_spec (Optional[VTraceFunction,dict]): A specification dict to construct a VTraceFunction or
                a VTraceFunction object to use.
            weight_pg (float): The coefficient used for the policy gradient loss term (L[PG]).
            weight_baseline (float): The coefficient used for the Value-function baseline term (L[V]).
            weight_entropy (float): The coefficient used for the entropy regularization term (L[E]).
                In the paper, values between 0.01 and 0.00005 are used via log-uniform search.
        """
        super(IMPALALossFunction, self).__init__(scope=kwargs.pop("scope", "impala-loss-func"), **kwargs)

        self.v_trace_function = VTraceFunction.from_spec(v_trace_spec)  # type: VTraceFunction

        self.weight_pg = weight_pg
        self.weight_baseline = weight_baseline
        self.weight_entropy = weight_entropy

    def _graph_fn_loss_per_item(self, action_probs, values, actions, rewards, terminals):
        """
        Calculates the loss per batch item (summed over all timesteps) using the formula described above in
        the docstring to this class.

        Args:
            action_probs (DataOp): The probabilities for all possible actions coming from the learner's policy (pi).
                Dimensions are: time x batch x data.
            values (DataOp): The state value estimates coming from baseline node of the learner's policy (pi).
                Dimensions are: time x batch x data.
            actions (DataOp): The actually taken actions. Dimensions are: time x batch x data.
            rewards (DataOp): The received rewards. Dimensions are: time x batch x data.
            terminals (DataOp): The observed terminal signals. Dimensions are: time x batch x data.

        Returns:
            SingleDataOp: The loss values per item in the batch, but summed over all timesteps.
        """
        # Let the v-trace  helper function calculate the v-trace values (vs) and the pg-advantages
        # (already multiplied by rho_t_pg): A = rho_t_pg * (rt + gamma*vt - V(t)).
        # Both vs and pg_advantages will block the gradient as they should be treated as constants by the gradient
        # calculator of this loss func.
        vs, pg_advantages = self.v_trace_function._graph_fn_calc_v_trace_values(TODO)

        if get_backend() == "tf":
            # The policy gradient loss.
            loss_pg = pg_advantages * tf.log(action_probs)  # TODO: get the probs only of the actions taken

            # The value-function baseline loss.
            loss_baseline = 0.5 * tf.square(x=tf.subtract(vs, values))

            # The entropy regularizer term.
            loss_entropy = - tf.reduce_sum(tf.multiply(action_probs, tf.log(action_probs)))

            return self.weight_pg * loss_pg + self.weight_baseline * loss_baseline + self.weight_entropy * loss_entropy
