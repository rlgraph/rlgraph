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

from rlgraph.utils.rlgraph_errors import RLGraphError


def get_action_adapter_type_from_distribution_type(distribution_type_str):
    """
    Args:
        distribution_type_str (str): The type (str) of the Distribution object, for which to return an appropriate
            ActionAdapter lookup-class string.

    Returns:
        str: The lookup-class string for an action-adapter.
    """
    # IntBox: Categorical.
    if distribution_type_str == "Categorical":
        return "categorical-distribution-adapter"
    elif distribution_type_str == "GumbelSoftmax":
        return "gumbel-softmax-distribution-adapter"
    # BoolBox: Bernoulli.
    elif distribution_type_str == "Bernoulli":
        return "bernoulli-distribution-adapter"
    # Continuous action space: Normal/Beta/etc. distribution.
    # Unbounded -> Normal distribution.
    elif distribution_type_str == "Normal":
        return "normal-distribution-adapter"
    # Bounded -> Beta.
    elif distribution_type_str == "Beta":
        return "beta-distribution-adapter"
    # Bounded -> Squashed Normal.
    elif distribution_type_str == "SquashedNormal":
        return "squashed-normal-distribution-adapter"
    else:
        raise RLGraphError("'{}' is an unknown Distribution type!".format(distribution_type_str))


def get_distribution_spec_from_action_adapter(action_adapter):
    action_adapter_type_str = type(action_adapter).__name__
    if action_adapter_type_str == "CategoricalDistributionAdapter":
        return dict(type="categorical")
    elif action_adapter_type_str == "GumbelSoftmaxDistributionAdapter":
        return dict(type="gumbel-softmax")
    elif action_adapter_type_str == "BernoulliDistributionAdapter":
        return dict(type="bernoulli")
    # TODO: What about multi-variate normal with non-trivial co-var matrices?
    elif action_adapter_type_str == "NormalDistributionAdapter":
        return dict(type="normal")
    elif action_adapter_type_str == "BetaDistributionAdapter":
        return dict(type="beta")
    elif action_adapter_type_str == "SquashedNormalDistributionAdapter":
        return dict(type="squashed-normal")
    elif action_adapter_type_str == "NormalMixtureDistributionAdapter":
        # TODO: MixtureDistribution is generic (any sub-distributions, but its AA is not (only supports mixture-Normal))
        return dict(type="mixture", _args=["multivariate-normal" for _ in range(action_adapter.size_mixture)])
    else:
        raise RLGraphError("'{}' is an unknown ActionAdapter type!".format(action_adapter_type_str))
