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

from yarl.components.common.splitter import Splitter
from yarl.components.common.merger import Merger
from yarl.components.common.synchronizable import Synchronizable
from yarl.components.common.decay_components import *
from yarl.components.common.noise_components import *
from yarl.components.common.fixed_loop import FixedLoop
from yarl.components.common.sampler import Sampler


DecayComponent.__lookup_classes__ = dict(
    lineardecay=LinearDecay,
    exponentialdecay=ExponentialDecay,
    polynomialdecay=PolynomialDecay
)

NoiseComponent.__lookup_classes__ = dict(
    constantnoise=ConstantNoise,
    gaussiannoise=GaussianNoise,
    ornsteinuhlenbeck=OrnsteinUhlenbeckNoise,
    ornsteinuhlenbecknoise=OrnsteinUhlenbeckNoise
)


__all__ = ["Splitter", "Merger",
           "Synchronizable",
           "DecayComponent", "LinearDecay", "PolynomialDecay", "ExponentialDecay",
           "NoiseComponent", "ConstantNoise", "GaussianNoise", "OrnsteinUhlenbeckNoise",
           "FixedLoop", "Sampler"]

