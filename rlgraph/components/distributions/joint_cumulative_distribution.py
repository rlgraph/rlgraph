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
from rlgraph.utils.ops import flatten_op

if get_backend() == "tf":
    import tensorflow as tf
elif get_backend() == "pytorch":
    import torch


class JointCumulativeDistribution(Distribution):
    """
    A joint cumulative distribution consisting of an arbitrarily nested container of n independent sub-distributions
    assumed to be all independent of each other, such that:
    For e.g. n=2 and random variables X and Y: P(X and Y) = P(X)*P(Y)) for all x and y.
    - Sampling returns a ContainerDataOp.
    - log_prob returns the sum of all single log prob terms (joint log prob).
    """
    def __init__(self, sub_distributions_spec, scope="joint-cumulative-distribution", **kwargs):
        """
        Args:
            sub_distributions_spec (Union[tuple,dict]): Tuple of dict (possibly nested) with the specifications of the
                single sub-distributions.
        """
        super(JointCumulativeDistribution, self).__init__(scope=scope, **kwargs)

        # Create the flattened sub-distributions and add them.
        flattened_sub_distributions = flatten_op(sub_distributions_spec)
        self.flattened_sub_distributions = \
            {k: Distribution.from_spec(s, scope="sub-distribution-{}".format(i))
             for i, (k, s) in enumerate(flattened_sub_distributions.items())
            }

        self.add_components(*list(self.flattened_sub_distributions.values()))

    # Flatten only alongside `self.flattened_sub_distributions`, not any further.
    @rlgraph_api(flatten_ops="flattened_sub_distributions", split_ops=True, add_auto_key_as_first_param=True)
    def _graph_fn_get_distribution(self, key, parameters):
        """
        Args:
            parameters (DataOpTuple): Tuple holding the mean and stddev parameters.
        """
        return self.flattened_sub_distributions[key].get_distribution(parameters)

    # Flatten only alongside `self.flattened_sub_distributions`, not any further.
    @graph_fn(flatten_ops="flattened_sub_distributions", split_ops=True, add_auto_key_as_first_param=True)
    def _graph_fn_sample_deterministic(self, key, distribution):
        return self.flattened_sub_distributions[key].sample_deterministic(distribution)

    # Flatten only alongside `self.flattened_sub_distributions`, not any further.
    @graph_fn(flatten_ops="flattened_sub_distributions")
    def _graph_fn_log_prob(self, distribution, values):
        all_log_probs = []
        for key, distr in distribution.items():
            all_log_probs.append(distr.log_prob(values[key]))

        if get_backend() == "tf":
            return tf.reduce_sum(tf.stack(all_log_probs, axis=0), axis=0)
        elif get_backend() == "pytorch":
            return torch.sum(torch.stack(all_log_probs, dim=0), dim=0)
