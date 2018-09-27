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

from functools import partial

from rlgraph.components.optimizers.horovod_optimizer import HorovodOptimizer
from rlgraph.components.optimizers.local_optimizers import *
from rlgraph.components.common.multi_gpu_synchronizer import MultiGpuSynchronizer
from rlgraph.components.optimizers.optimizer import Optimizer


Optimizer.__lookup_classes__ = dict(
    horovod=HorovodOptimizer,
    #multigpu=MultiGpuSynchronizer,
    #multigpusync=MultiGpuSynchronizer,
    # LocalOptimizers.
    gradientdescent=GradientDescentOptimizer,
    adagrad=AdagradOptimizer,
    adadelta=AdadeltaOptimizer,
    adam=AdamOptimizer,
    nadam=NadamOptimizer,
    sgd=SGDOptimizer,
    rmsprop=RMSPropOptimizer
)

# The default Optimizer to use if a spec is None and no args/kwars are given.
Optimizer.__default_constructor__ = partial(GradientDescentOptimizer, learning_rate=0.0001)

__all__ = ["Optimizer", "LocalOptimizer", "MultiGpuSynchronizer"] + \
          list(set(map(lambda x: x.__name__, Optimizer.__lookup_classes__.values())))
