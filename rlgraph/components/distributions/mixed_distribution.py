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
from rlgraph.utils.decorators import rlgraph_api, graph_fn
from rlgraph.components.distributions.distribution import Distribution

if get_backend() == "tf":
    import tensorflow as tf
    import tensorflow_probability as tfp
elif get_backend() == "pytorch":
    import torch


class MixedDistribution(Distribution):
    """
    A mixed distribution of n sub-distribution components and a categorical which determines,
    from which sub-distribution we sample.
    """
    def __init__(self, *sub_distributions, **kwargs):
        """
        Args:
            sub_distributions (List[Union[string,Distribution]]): The type-strings or actual Distribution objects
                that define the n sub-distributions of this MixedDistribution.
        """
        super(MixedDistribution, self).__init__(scope=kwargs.pop("scope", "mixed-distribution"), **kwargs)

        self.sub_distributions = []
        for s in sub_distributions:
            if isinstance(s, str):
                assert s in ["normal", "beta"],\
                    "ERROR: MixedDistribution does not accept '{}' as sub-distribution type!".format(s)
                self.sub_distributions.append(s)
            elif isinstance(s, Normal):
                self.sub_distributions.append("normal")
            elif isinstance(s, Beta):
                self.sub_distributions.append("beta")
            else:
                raise RLGrapahError(
                    "MixedDistribution does not accept '{}' as sub-distribution type!".format(type(s).__name__)
                )
        #self.num_distributions = len(self.sub_distributions)

        # list of ints indicating how many parameters are needed for each sub-distribution
        self.num_parameters = list()

    def check_input_spaces(self, input_spaces, action_space=None):
        # Must be a Tuple of len 2 (loc and scale).
        in_space = input_spaces["parameters"]
        # Make sure input parameters has an even last rank for splitting into mean/stddev parameter values.
        #assert in_space.shape[-1] % 2 == 0,\
        #    "ERROR: `parameters` in_space must have an even numbered last rank (mean/stddev split)!"

    @rlgraph_api
    def _graph_fn_get_distribution(self, parameters):
        """
        Instead of directly parameterizing a fixed distribution, select first the distribution
        to use via our Categorical and the gievn parameters. Then feed the rest of the parameters
        and the selected distribution through its own get_distribution API.

        Args:
            parameters (DataOp): The parameters to use for parameterization of a Distribution DataOp.

        Returns:
            DataOp: The ready-to-be-sampled distribution.
        """
        if get_backend() == "tf":
            components = []
            num_or_size_splits = [len(self.sub_distributions)]
            for s in self.sub_distributions:
                num_or_size_splits.append()
                components.append()
            probabilities, rest = tf.split(parameters, num_or_size_splits=[self.num_distributions, ], axis=-1)

            return tfp.distributions.Mixture(
                cat=tfp.Categorical(probs=probabilities),
                components=components
            )

    @graph_fn
    def _graph_fn_sample_deterministic(self, distribution):
            # TODO:
            return distribution.mean()
