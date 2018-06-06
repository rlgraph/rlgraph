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


LocalOptimizer.__lookup_classes__ = dict(
    gradient_descent=GradientDescentOptimizer,
    adagrad=AdagradOptimizer,
    adadelta=AdadeltaOptimizer,
    adam=AdamOptimizer,
    nadam=NadamOptimizer,
    sgd=SGDOptimizer,
    rmsprop=RMSPropOptimizer,
    horovod=HorovodOptimizer
)

__all__ = ["Optimizer", "LocalOptimizer", "GradientDescentOptimizer", "AdagradOptimizer",
           "AdadeltaOptimizer", "AdamOptimizer", "NadamOptimizer", "SGDOptimizer",
           "RMSPropOptimizer"]

