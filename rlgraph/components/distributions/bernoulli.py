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
from rlgraph.utils.util import convert_dtype
from rlgraph.utils.decorators import rlgraph_api, graph_fn
from rlgraph.components.distributions.distribution import Distribution
from rlgraph.components.action_adapters import CategoricalDistributionAdapter

if get_backend() == "tf":
    import tensorflow_probability as tfp
elif get_backend() == "pytorch":
    import torch


class Bernoulli(Distribution):
    """
    A Bernoulli distribution object defined by a single value p, the probability for True (rather than False).
    """
    def __init__(self, scope="bernoulli", **kwargs):
        super(Bernoulli, self).__init__(scope=scope, **kwargs)

    @rlgraph_api
    def _graph_fn_get_distribution(self, parameters):
        """
        Args:
            parameters (DataOp): The p value (probability that distribution returns True).
        """
        if get_backend() == "tf":
            return tfp.distributions.Bernoulli(probs=parameters, dtype=convert_dtype("bool"))
        elif get_backend() == "pytorch":
            return torch.distributions.Bernoulli(probs=parameters)

    @graph_fn
    def _graph_fn_sample_deterministic(self, distribution):
        return distribution.prob(True) >= 0.5

    def get_action_adapter_type(self):
        return CategoricalDistributionAdapter
