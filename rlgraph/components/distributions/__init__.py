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

from rlgraph.components.distributions.distribution import Distribution
from rlgraph.components.distributions.bernoulli import Bernoulli
from rlgraph.components.distributions.beta import Beta
from rlgraph.components.distributions.categorical import Categorical
from rlgraph.components.distributions.mixture_distribution import MixtureDistribution
from rlgraph.components.distributions.multivariate_normal import MultivariateNormal
from rlgraph.components.distributions.normal import Normal
from rlgraph.components.distributions.squashed_normal import SquashedNormal

Distribution.__lookup_classes__ = dict(
    bernoulli=Bernoulli,
    bernoullidistribution=Bernoulli,
    categorical=Categorical,
    categoricaldistribution=Categorical,
    gaussian=Normal,
    gaussiandistribution=Normal,
    mixed=MixtureDistribution,
    mixeddistribution=MixtureDistribution,
    multivariatenormal=MultivariateNormal,
    multivariategaussian=MultivariateNormal,
    normaldistribution=Normal,
    beta=Beta,
    betadistribution=Beta,
    squashed=SquashedNormal,
    squashednormal=SquashedNormal,
    squashednormaldistribution=SquashedNormal,
)

__all__ = ["Distribution"] + list(set(map(lambda x: x.__name__, Distribution.__lookup_classes__.values())))

