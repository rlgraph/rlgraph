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
from rlgraph.components.action_adapters import SquashedNormalAdapter
from rlgraph.components.distributions.distribution import Distribution
from rlgraph.spaces import Tuple, FloatBox
from rlgraph.spaces.space_utils import sanity_check_space
from rlgraph.utils.decorators import rlgraph_api, graph_fn
from rlgraph.utils.util import SMALL_NUMBER

if get_backend() == "tf":
    import tensorflow as tf
    import tensorflow_probability as tfp
elif get_backend() == "pytorch":
    import torch


class SquashedNormal(Distribution):
    """
    A Squashed with tanh Normal distribution object defined by a tuple: mean, standard deviation.
    """
    def __init__(self, scope="squashed-normal", low=-1.0, high=1.0, **kwargs):
        # Do not flatten incoming DataOps as we need more than one parameter in our parameterize graph_fn.
        assert low < high
        self.low = low
        self.high = high
        super(SquashedNormal, self).__init__(scope=scope, **kwargs)

    def check_input_spaces(self, input_spaces, action_space=None):
        # Must be a Tuple of len 2 (loc and scale).
        in_space = input_spaces["parameters"]
        sanity_check_space(in_space, allowed_types=[Tuple])
        assert len(in_space) == 2, "ERROR: Expected Tuple of len=2 as input Space to SquashedNormal!"
        sanity_check_space(in_space[0], allowed_types=[FloatBox])
        sanity_check_space(in_space[1], allowed_types=[FloatBox])

    @rlgraph_api
    def sample_and_log_prob(self, parameters, deterministic=True):
        distribution = self.get_distribution(parameters)
        scaled_actions, log_prob = self._graph_fn_sample_and_log_prob(distribution, deterministic)
        return scaled_actions, log_prob

    @rlgraph_api
    def _graph_fn_get_distribution(self, parameters):
        if get_backend() == "tf":
            return tfp.distributions.Normal(loc=parameters[0], scale=parameters[1])
        elif get_backend() == "pytorch":
            return torch.distributions.Normal(parameters[0], parameters[1])

    @graph_fn
    def _graph_fn_squash(self, raw_values):
        if get_backend() == "tf":
            return (tf.tanh(raw_values) + 1.0) / 2.0 * (self.high - self.low) + self.low
        elif get_backend() == "pytorch":
            return (torch.tanh(raw_values) + 1.0) / 2.0 * (self.high - self.low) + self.low

    @graph_fn
    def _graph_fn_unsquash(self, values):
        if get_backend() == "tf":
            return tf.atanh((values - self.low) / (self.high - self.low) * 2.0 - 1.0)
        elif get_backend() == "tf":
            return torch.atanh((values - self.low) / (self.high - self.low) * 2.0 - 1.0)

    @graph_fn
    def _graph_fn_sample_deterministic(self, distribution):
        mean = None
        if get_backend() == "tf":
            mean = distribution.mean()
        elif get_backend() == "pytorch":
            mean = distribution.mean
        return self._graph_fn_squash(mean)

    @graph_fn
    def _graph_fn_sample_stochastic(self, distribution):
        if get_backend() == "tf":
            return self._graph_fn_squash(distribution.sample(seed=self.seed))
        elif get_backend() == "pytorch":
            return self._graph_fn_squash(distribution.sample())

    @graph_fn
    def _graph_fn_log_prob(self, distribution, values):
        log_prob = distribution.log_prob(value=self._graph_fn_unsquash(values))
        if get_backend() == "tf":
            log_prob -= tf.reduce_sum(tf.log(1 - values ** 2 + SMALL_NUMBER), axis=1, keepdims=True)
        elif get_backend() == "pytorch":
            log_prob -= torch.sum(tf.log(1 - values ** 2 + SMALL_NUMBER), axis=1, keepdims=True)
        return log_prob

    @graph_fn
    def _graph_fn_entropy(self, distribution):
        return distribution.entropy()

    @graph_fn
    def _graph_fn_kl_divergence(self, distribution, distribution_b):
        if get_backend() == "tf":
            return tf.no_op()
        elif get_backend() == "pytorch":
            return None

    @graph_fn
    def _graph_fn_sample_and_log_prob(self, distribution, deterministic):
        if get_backend() == "tf":
            raw_action = tf.cond(
                pred=deterministic,
                true_fn=lambda: distribution.mean(),
                false_fn=lambda: distribution.sample()
            )
            action = tf.tanh(raw_action)
            log_prob = distribution.log_prob(raw_action)
            log_prob -= tf.reduce_sum(tf.log(1 - action ** 2 + SMALL_NUMBER), axis=1, keepdims=True)
            scaled_action = (action + 1) / 2 * (self.high - self.low) + self.low
            return scaled_action, log_prob
        elif get_backend() == "pytorch":
            if deterministic:
                raw_action = distribution.mean()
            else:
                raw_action = distribution.sample()
            # TODO: ...
            raise NotImplementedError

    def get_action_adapter_type(self):
        return SquashedNormalAdapter
