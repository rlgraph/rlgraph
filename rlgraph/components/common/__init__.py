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

from rlgraph.components.common.container_splitter import ContainerSplitter
from rlgraph.components.common.dict_merger import DictMerger
from rlgraph.components.common.synchronizable import Synchronizable
from rlgraph.components.common.decay_components import DecayComponent, LinearDecay, PolynomialDecay, ExponentialDecay, \
    ConstantDecay
from rlgraph.components.common.noise_components import NoiseComponent, ConstantNoise, GaussianNoise, \
    OrnsteinUhlenbeckNoise
from rlgraph.components.common.fixed_loop import FixedLoop
from rlgraph.components.common.sampler import Sampler
from rlgraph.components.common.slice import Slice
from rlgraph.components.common.staging_area import StagingArea
from rlgraph.components.common.repeater_stack import RepeaterStack
from rlgraph.components.common.batch_splitter import BatchSplitter
from rlgraph.components.common.multi_gpu_synchronizer import MultiGpuSynchronizer


DecayComponent.__lookup_classes__ = dict(
    lineardecay=LinearDecay,
    constantdecay=ConstantDecay,
    exponentialdecay=ExponentialDecay,
    polynomialdecay=PolynomialDecay
)
DecayComponent.__default_constructor__ = LinearDecay

NoiseComponent.__lookup_classes__ = dict(
    constantnoise=ConstantNoise,
    gaussiannoise=GaussianNoise,
    ornsteinuhlenbeck=OrnsteinUhlenbeckNoise,
    ornsteinuhlenbecknoise=OrnsteinUhlenbeckNoise
)
NoiseComponent.__default_constructor__ = GaussianNoise


__all__ = ["ContainerSplitter", "DictMerger",
           "Synchronizable", "RepeaterStack", "Slice",
           "DecayComponent", "ConstantDecay", "LinearDecay", "PolynomialDecay", "ExponentialDecay",
           "NoiseComponent", "ConstantNoise", "GaussianNoise", "OrnsteinUhlenbeckNoise",
           "FixedLoop", "Sampler", "BatchSplitter", "MultiGpuSynchronizer"]

