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

from rlgraph.components.component import Component
from rlgraph.components.helpers import SequenceHelper
from rlgraph.components.helpers.clipping import Clipping
from rlgraph.utils.decorators import rlgraph_api


class GeneralizedAdvantageEstimation(Component):
    """
    A Helper Component that contains a graph_fn to generalized advantage estimation (GAE) [1].

    [1] High-Dimensional Continuous Control Using Generalized Advantage Estimation - Schulman et al.
    - 2015 (https://arxiv.org/abs/1506.02438)
    """

    def __init__(self, gae_lambda=1.0, discount=1.0, clip_rewards=0.0,
                 scope="generalized-advantage-estimation", **kwargs):
        """
        Args:
            gae_lambda (float): GAE-lambda. See paper for details.
            discount (float): Discount gamma.
        """
        super(GeneralizedAdvantageEstimation, self).__init__(scope=scope, **kwargs)
        self.gae_lambda = gae_lambda
        self.discount = discount
        self.sequence_helper = SequenceHelper()
        self.clipping = Clipping(clip_value=clip_rewards)

        self.add_components(self.sequence_helper, self.clipping)

    @rlgraph_api
    def calc_td_errors(self, baseline_values, rewards, terminals, sequence_indices):
        """
        Returns 1-step TD Errors (delta = r + gamma V(s') - V(s)). Clips rewards if specified.

        Args:
            baseline_values (DataOp): Baseline predictions V(s).
            rewards (DataOp): Rewards in sample trajectory.
            terminals (DataOp): Terminals in sample trajectory.
            sequence_indices (DataOp): Int indices denoting sequences (which may be non-terminal episode fragments
                from multiple environments.

        Returns:
            1-Step TD Errors for the sequence.
        """
        rewards = self.clipping.clip_if_needed(rewards)

        # Next, we need to set the next value after the end of each sub-sequence to 0/its prior value
        # depending on terminal, then compute 1-step TD-errors: delta = r[:] + gamma * v[1:] - v[:-1]
        # -> where len(v) = len(r) + 1 b/c v contains the extra (bootstrapped) last value.
        # Terminals indicate boot-strapping. Sequence indices indicate episode fragments in case of a multi-environment.
        deltas = self.sequence_helper.bootstrap_values(
            rewards, baseline_values, terminals, sequence_indices, self.discount
        )
        return deltas

    @rlgraph_api
    def calc_gae_values(self, baseline_values, rewards, terminals, sequence_indices):
        """
        Returns advantage values based on GAE. Clips rewards if specified.

        Args:
            baseline_values (DataOp): Baseline predictions V(s).
            rewards (DataOp): Rewards in sample trajectory.
            terminals (DataOp): Terminals in sample trajectory.
            sequence_indices (DataOp): Int indices denoting sequences (which may be non-terminal episode fragments
                from multiple environments.
        Returns:
            PG-advantage values used for training via policy gradient with baseline.
        """
        deltas = self.calc_td_errors(baseline_values, rewards, terminals, sequence_indices)

        gae_discount = self.gae_lambda * self.discount
        # Apply gae discount to each sub-sequence.
        # Note: sequences are indicated by sequence indices, which may not be terminal.
        gae_values = self.sequence_helper.reverse_apply_decays_to_sequence(deltas, sequence_indices, gae_discount)
        return gae_values
