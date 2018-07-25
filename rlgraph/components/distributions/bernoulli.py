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
from rlgraph.utils import util
from rlgraph.components.distributions.distribution import Distribution

if get_backend() == "tf":
    import tensorflow as tf


class Bernoulli(Distribution):
    """
    A Bernoulli distribution object defined by a single value p, the probability for True (rather than False).
    """
    def __init__(self, scope="bernoulli", **kwargs):
        super(Bernoulli, self).__init__(scope=scope, **kwargs)

    def _graph_fn_get_distribution(self, prob):
        """
        Args:
            prob (DataOp): The p value (probability that distribution returns True).
        """
        # TODO: Move this into some sort of NN-output-cleanup component. This is repetitive stuff.
        ## Clip raw_input between 0 and 1 to make it interpretable as a probability.
        #p = tf.sigmoid(x=flat_input)
        ## Clip to avoid 0.0 or 1.0 probabilities (adds numerical stability).
        #p = tf.clip_by_value(p, clip_value_min=SMALL_NUMBER, clip_value_max=(1.0 - SMALL_NUMBER))
        if get_backend() == "tf":
            return tf.distributions.Bernoulli(probs=prob, dtype=util.dtype("bool"))

    def _graph_fn_sample_deterministic(self, distribution):
        return distribution.prob(True) >= 0.5

