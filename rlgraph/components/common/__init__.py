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

from rlgraph.components.common.batch_apply import BatchApply
from rlgraph.components.common.batch_splitter import BatchSplitter
from rlgraph.components.common.container_merger import ContainerMerger
# TODO: Obsoleted classes.
from rlgraph.components.common.decay_components import DecayComponent, ConstantDecay
from rlgraph.components.common.multi_gpu_synchronizer import MultiGpuSynchronizer
from rlgraph.components.common.noise_components import NoiseComponent, ConstantNoise, GaussianNoise, \
    OrnsteinUhlenbeckNoise
from rlgraph.components.common.repeater_stack import RepeaterStack
from rlgraph.components.common.sampler import Sampler
from rlgraph.components.common.slice import Slice
from rlgraph.components.common.staging_area import StagingArea
from rlgraph.components.common.synchronizable import Synchronizable
from rlgraph.components.common.time_dependent_parameters import TimeDependentParameter, Constant, \
    LinearDecay, PolynomialDecay, ExponentialDecay

TimeDependentParameter.__lookup_classes__ = dict(
    parameter=TimeDependentParameter,
    constant=Constant,
    constantparameter=Constant,
    constantdecay=Constant,
    lineardecay=LinearDecay,
    polynomialdecay=PolynomialDecay,
    exponentialdecay=ExponentialDecay
)
TimeDependentParameter.__default_constructor__ = Constant

# TODO: Obsoleted classes.
DecayComponent.__lookup_classes__ = dict(
    decay=DecayComponent
)

NoiseComponent.__lookup_classes__ = dict(
    noise=NoiseComponent,
    constantnoise=ConstantNoise,
    gaussiannoise=GaussianNoise,
    ornsteinuhlenbeck=OrnsteinUhlenbeckNoise,
    ornsteinuhlenbecknoise=OrnsteinUhlenbeckNoise
)
NoiseComponent.__default_constructor__ = GaussianNoise


__all__ = ["BatchApply", "ContainerMerger",
           "Synchronizable", "RepeaterStack", "Slice",
           "Sampler", "BatchSplitter", "MultiGpuSynchronizer"] + \
          list(set(map(lambda x: x.__name__, list(TimeDependentParameter.__lookup_classes__.values()) +
                       list(DecayComponent.__lookup_classes__.values()) +
                       list(NoiseComponent.__lookup_classes__.values()))))
