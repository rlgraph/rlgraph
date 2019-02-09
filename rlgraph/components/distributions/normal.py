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
elif get_backend() == "pytorch":
    import torch


class Normal(Distribution):
    """
    A Gaussian Normal distribution object defined by a tuple: mean, variance,
    which is the same as "loc_and_scale".
    """
    def __init__(self, scope="normal", **kwargs):
        # Do not flatten incoming DataOps as we need more than one parameter in our parameterize graph_fn.
        super(Normal, self).__init__(scope=scope, **kwargs)

    def check_input_spaces(self, input_spaces, action_space=None):
        # Must be a Tuple of len 2 (loc and scale).
        in_space = input_spaces["parameters"]
        # Make sure input parameters has an even last rank for splitting into mean/stddev parameter values.
        assert in_space.shape[-1] % 2 == 0, "ERROR: `parameters` in_space must have an even numbered last rank!"

    @rlgraph_api
    def _graph_fn_get_distribution(self, parameters):
        if get_backend() == "tf":
            mean, stddev = tf.split(parameters, num_or_size_splits=2, axis=-1)
            return tf.distributions.Normal(loc=mean, scale=stddev)
        elif get_backend() == "pytorch":
            mean, stddev = torch.split(parameters, 2, dim=-1)
            return torch.distributions.Normal(mean, stddev)

    @graph_fn
    def _graph_fn_sample_deterministic(self, distribution):
            return distribution.mean()
