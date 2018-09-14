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
from rlgraph.components.distributions.distribution import Distribution
from rlgraph.spaces import Tuple

if get_backend() == "tf":
    import tensorflow as tf
elif get_backend() == "pytorch":
    import torch


class Beta(Distribution):
    """
    A Beta distribution is defined on the interval [0, 1] and parameterized by shape parameters
    alpha and beta (also called concentration parameters).
    """
    def __init__(self, scope="beta", **kwargs):
        # Do not flatten incoming DataOps as we need more than one parameter in our parameterize graph_fn.
        super(Beta, self).__init__(scope=scope, **kwargs)

    def check_input_spaces(self, input_spaces, action_space=None):
        # Must be a Tuple of len 2 (loc and scale).
        in_space = input_spaces["concentration_parameters"]
        assert isinstance(in_space, Tuple) and len(in_space) == 2,\
            "ERROR: {} (Distribution) ({}) needs an incoming Tuple with len=2!".format(type(self).__name__,
                                                                                       self.name)

    def _graph_fn_get_distribution(self, parameters):
        if get_backend() == "tf":
            return tf.distributions.Beta(
                concentration0=parameters[0],
                concentration1=parameters[1]
            )
        elif get_backend() == "pytorch":
            return torch.distributions.Beta(parameters[0], parameters[1])

    def _graph_fn_sample_deterministic(self, distribution):
        return distribution.mean()
