# Copyright 2018 The RLGraph-Project, All Rights Reserved.
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

from rlgraph.components.helpers.v_trace_function import VTraceFunction
from rlgraph.components.loss_functions import LossFunction


class IMPALALossFunction(LossFunction):
    """
    The IMPALA loss function based on v-trace off-policy policy gradient corrections, described in detail in [1].

    [1] IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures - Espeholt, Soyer,
        Munos et al. - 2018 (https://arxiv.org/abs/1802.01561)
    """
    def __init__(self, v_trace_spec=None, **kwargs):
        super(IMPALALossFunction, self).__init__(scope=kwargs.pop("scope", "impala-loss-func"), **kwargs)

        self.v_trace = VTraceFunction.from_spec(v_trace_spec)

    def _graph_fn_loss_per_item(self, logits, advantages, actions):
        """
        Calculates the loss per batch item (summed over all timesteps) based on: Advantages, raw policy logits
        and actually selected actions from some policy.

        Args:
            logits (DataOp): The raw logits (before softmaxing) coming from the policy. Dimensions are:
                batch x time x data.
            advantages (DataOp): The advantage values coming from the policy. Dimensions are:
                batch x time x data.
            actions (DataOp): The actually taken actions. Dimensions are:
                batch x time x data.

        Returns:
            DataOp: The loss values per item in the batch, but summed over all timesteps.
        """
        pass
