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

    def __init__(self, **kwargs):
        super(VTraceFunction, self).__init__(scope=kwargs.pop("scope", "v-trace-function"), **kwargs)

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

