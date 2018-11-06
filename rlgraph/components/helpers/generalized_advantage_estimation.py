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

from rlgraph.components import Component
from rlgraph.components.helpers import SequenceHelper
from rlgraph.utils.decorators import rlgraph_api


class GeneralizedAdvantageEstimation(Component):
    """
    A Helper Component that contains a graph_fn to generalized advantage estimation (GAE) [1].

    [1] High-Dimensional Continuous Control Using Generalized Advantage Estimation - Schulman et al.
    - 2015 (https://arxiv.org/abs/1506.02438)
    """

    def __init__(self, gae_lambda=1.0, discount=1.0, device="/device:CPU:0",
                 scope="generalized-advantage-estimation", **kwargs):
        """
        Args:
            gae_lambda (float): GAE-lambda. See paper for details.
            discount (float): Discount gamma.
        """
        super(GeneralizedAdvantageEstimation, self).__init__(device=device, scope=scope, **kwargs)
        self.gae_lambda = gae_lambda
        self.discount = discount
        self.sequence_helper = SequenceHelper()
        self.add_components(self.sequence_helper)

    @rlgraph_api(must_be_complete=False)
    def _graph_fn_calc_gae_values(self, baseline_values, rewards, terminals):
        """
        Returns advantage values based on GAE.

        Args:
            baseline_values (DataOp): Baseline predictions V(s).
            rewards (DataOp): Rewards in sample trajectory.
            terminals (DataOp): Terminals in sample trajectory.

        Returns:
            PG-advantage values used for training via policy gradient with baseline.
        """
        gae_discount = self.gae_lambda * self.discount

        # Next, we need to set the next value after the end of each sub-sequence to 0/its prior value
        # depending on terminal, then compute deltas = r + y * v[1:] - v[:-1]
        deltas = self.sequence_helper.bootstrap_values(rewards, baseline_values, terminals, self.discount)

        # Apply gae discount to each sub-sequence.
        return self.sequence_helper.reverse_apply_decays_to_sequence(deltas, terminals, gae_discount)
