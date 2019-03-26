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
from rlgraph.components.distributions.categorical import Categorical
from rlgraph.components.distributions.distribution import Distribution
from rlgraph.utils.decorators import rlgraph_api, graph_fn
from rlgraph.utils.rlgraph_errors import RLGraphError

if get_backend() == "tf":
    import tensorflow_probability as tfp
elif get_backend() == "pytorch":
    pass


class MixtureDistribution(Distribution):
    """
    A mixed distribution of n sub-distribution components and a categorical which determines,
    from which sub-distribution we sample.
    """
    def __init__(self, *sub_distributions, **kwargs):
        """
        Args:
            sub_distributions (List[Union[string,Distribution]]): The type-strings or actual Distribution objects
                that define the n sub-distributions of this MixtureDistribution.
        """
        super(MixtureDistribution, self).__init__(scope=kwargs.pop("scope", "mixed-distribution"), **kwargs)

        self.sub_distributions = []
        for i, s in enumerate(sub_distributions):
            if isinstance(s, str):
                self.sub_distributions.append(Distribution.from_spec(
                    {"type": s, "scope": "sub-distribution-{}".format(i)}
                ))
            else:
                self.sub_distributions.append(Distribution.from_spec(s))

        self.categorical = Categorical(scope="main-categorical")

        self.add_components(self.categorical, *self.sub_distributions)

    def check_input_spaces(self, input_spaces, action_space=None):
        # Must be a Dict with keys: 'categorical', 'parameters0', 'parameters1', etc...
        in_space = input_spaces["parameters"]

        assert "categorical" in in_space, "ERROR: in_space for Mixed needs parameter key: 'categorical'!"

        for i, s in enumerate(self.sub_distributions):
            sub_space = in_space.get("parameters{}".format(i))
            if sub_space is None:
                raise RLGraphError("ERROR: in_space for Mixed needs parameter key: 'parameters{}'!".format(i))

    @rlgraph_api
    def _graph_fn_get_distribution(self, parameters):
        """
        Args:
            parameters (DataOpDict): The parameters to use for parameterizations of the different sub-distributions
                including the main-categorical one. Keys must be "categorical", "parameters0", "parameters1", etc..

        Returns:
            DataOp: The ready-to-be-sampled mixed distribution.
        """
        if get_backend() == "tf":
            components = []
            for i, s in enumerate(self.sub_distributions):
                components.append(s.get_distribution(parameters["parameters{}".format(i)]))

            return tfp.distributions.Mixture(
                cat=self.categorical.get_distribution(parameters["categorical"]),
                components=components
            )

    @rlgraph_api
    def entropy(self, parameters):
        """
        Not implemented for Mixed. Raise error for now (safer than returning wrong values).
        """
        raise NotImplementedError

    @graph_fn
    def _graph_fn_sample_deterministic(self, distribution):
        if get_backend() == "tf":
            return distribution.mean()
        elif get_backend() == "pytorch":
            return distribution.mean
