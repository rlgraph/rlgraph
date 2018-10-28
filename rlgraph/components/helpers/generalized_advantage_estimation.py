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
from rlgraph.components import Component
from rlgraph.components.helpers import SequenceHelper
from rlgraph.utils.decorators import rlgraph_api

if get_backend() == "tf":
    import tensorflow as tf
    import trfl


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
    def _graph_fn_calc_gae_values(self, rewards, terminals):
        """
        Returns advantage values based on GAE.

        Args:
            rewards (DataOp): Rewards in sample trajectory.
            terminals (DataOp): Termnials in sample trajectory.

        Returns:
            PG-advantage values used for training via policy gradient with baseline.
        """
        if get_backend() == "tf":
            gae_discount = self.gae_lambda * self.discount
            sequence_lengths = self.sequence_helper.calc_sequence_lengths(terminals)

            # deltas = rewards + self.discount * baseline_values[1:]
            # TODO call TRFL
            advantages = None

            return advantages
