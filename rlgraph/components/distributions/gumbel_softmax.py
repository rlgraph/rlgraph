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
from rlgraph.components.action_adapters import GumbelSoftmaxAdapter
from rlgraph.components.distributions.distribution import Distribution
from rlgraph.utils import util
from rlgraph.utils.decorators import rlgraph_api, graph_fn

if get_backend() == "tf":
    import tensorflow as tf
    import tensorflow_probability as tfp
elif get_backend() == "pytorch":
    import torch


class GumbelSoftmax(Distribution):
    """
    The Gumbel Softmax distribution is also known as a relaxed one-hot categorical or concrete distribution.

    Gumbel Softmax: https://arxiv.org/abs/1611.01144

    Concrete: https://arxiv.org/abs/1611.00712
    """
    def __init__(self, scope="gumbel-softmax", temperature=0.1, **kwargs):
        """

        Args:
            temperature (float): Temperature parameter. For low temperatures, the expected value approaches
                a categorical random variable. For high temperatures, the expected value approaches a uniform
                distribution.
        """
        assert 0.0 < temperature < 1.0
        self.temperature = temperature
        super(GumbelSoftmax, self).__init__(scope=scope, **kwargs)

    @rlgraph_api
    def _graph_fn_get_distribution(self, parameters):
        if get_backend() == "tf":
            return tfp.distributions.RelaxedOneHotCategorical(temperature=self.temperature, probs=parameters)
        elif get_backend() == "pytorch":
            return torch.distributions.RelaxedOneHotCategorical(temperature=self.temperature, probs=parameters)

    @graph_fn
    def _graph_fn_sample_deterministic(self, distribution):
        if get_backend() == "tf":
            return tf.argmax(input=distribution._distribution.probs, axis=-1, output_type=util.convert_dtype("int"))
        elif get_backend() == "pytorch":
            return torch.argmax(distribution.probs, dim=-1).int()

    @graph_fn
    def _graph_fn_sample_stochastic(self, distribution):
        if get_backend() == "tf":
            return distribution.sample(seed=self.seed)
        elif get_backend() == "pytorch":
            return distribution.sample()

    @graph_fn
    def _graph_fn_log_prob(self, distribution, values):
        if get_backend() == "tf":
            return distribution.log_prob(values)
        elif get_backend() == "pytorch":
            return distribution.log_prob(values)

    @graph_fn
    def _graph_fn_entropy(self, distribution):
        return distribution.entropy()

    @graph_fn
    def _graph_fn_kl_divergence(self, distribution, distribution_b):
        if get_backend() == "tf":
            return tf.no_op()
        elif get_backend() == "pytorch":
            return None

    def get_action_adapter_type(self):
        return GumbelSoftmaxAdapter

