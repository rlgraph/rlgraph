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
from rlgraph.components.distributions.distribution import Distribution
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
    def __init__(self, scope="gumbel-softmax", temperature=1.0, **kwargs):
        """

        Args:
            temperature (float): Temperature parameter. For low temperatures, the expected value approaches
                a categorical random variable. For high temperatures, the expected value approaches a uniform
                distribution.
        """
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
        """
        Returns the argmax (int) of a relaxed one-hot vector. See `_graph_fn_sample_stochastic` for details.
        """
        if get_backend() == "tf":
            # Cast to float again because this is called rom a tf.cond where the other option calls a stochastic
            # sample returning a float.
            argmax = tf.argmax(input=distribution._distribution.probs, axis=-1, output_type=tf.int32)
            sample = tf.cast(argmax, dtype=tf.float32)
            # Argmax turns (?, n) into (?,), not (?, 1)
            # TODO: What if we have a time rank as well?
            if len(sample.shape) == 1:
                sample = tf.expand_dims(sample, -1)
                return sample
        elif get_backend() == "pytorch":
            # TODO: keepdims?
            return torch.argmax(distribution.probs, dim=-1).int()

    @graph_fn
    def _graph_fn_sample_stochastic(self, distribution):
        """
        Returns a relaxed one-hot vector representing a quasi-one-hot action.
        To get the actual int action, one would have to take the argmax over
        these output values. However, argmax would break the differentiability and should
        thus only be used right before applying the action in e.g. an env.
        """
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

