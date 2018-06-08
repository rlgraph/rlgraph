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

from yarl.components.optimizers.horovod_optimizer import HorovodOptimizer
from yarl.components.optimizers.local_optimizers import *
from yarl.components.optimizers.optimizer import Optimizer


Optimizer.__lookup_classes__ = dict(
    horovod=HorovodOptimizer
)
# The default Optimizer to use if a spec is None and no args/kwars are given.
Optimizer.__default_object__ = SGDOptimizer(learning_rate=0.0001)

LocalOptimizer.__lookup_classes__ = dict(
    gradientdescent=GradientDescentOptimizer,
    adagrad=AdagradOptimizer,
    adadelta=AdadeltaOptimizer,
    adam=AdamOptimizer,
    nadam=NadamOptimizer,
    sgd=SGDOptimizer,
    rmsprop=RMSPropOptimizer
)

__all__ = ["Optimizer", "LocalOptimizer", "HorovodOptimizer",
           "GradientDescentOptimizer", "SGDOptimizer",
           "AdagradOptimizer", "AdadeltaOptimizer", "AdamOptimizer", "NadamOptimizer",
           "RMSPropOptimizer"]
