# Copyright 2018 The YARL-Project, All Rights Reserved.
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

from yarl.components.distributions.distribution import Distribution
from yarl.components.distributions.bernoulli import Bernoulli
from yarl.components.distributions.beta import Beta
from yarl.components.distributions.categorical import Categorical
from yarl.components.distributions.normal import Normal

Distribution.__lookup_classes__ = dict(
    bernoulli=Bernoulli,
    categorical=Categorical,
    normaldistribution=Normal,
    gaussian=Normal,
    beta=Beta
)

__all__ = ["Distribution", "Bernoulli", "Categorical", "Normal", "Beta"]

