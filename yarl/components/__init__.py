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

from yarl.utils.util import default_dict

# Core.
from yarl.components.component import Component
# Component child-classes.
from yarl.components.common import *
from yarl.components.distributions import *
from yarl.components.explorations import *
from yarl.components.layers import *
from yarl.components.loss_functions import *
from yarl.components.memories import *
from yarl.components.neural_networks import *
from yarl.components.optimizers import *

Component.__lookup_classes__ = dict()
# Add all specific sub-classes to this one.
default_dict(Component.__lookup_classes__, Distribution.__lookup_classes__)
default_dict(Component.__lookup_classes__, Layer.__lookup_classes__)
default_dict(Component.__lookup_classes__, Stack.__lookup_classes__)
default_dict(Component.__lookup_classes__, LossFunction.__lookup_classes__)
default_dict(Component.__lookup_classes__, Memory.__lookup_classes__)
default_dict(Component.__lookup_classes__, NeuralNetwork.__lookup_classes__)
default_dict(Component.__lookup_classes__, Optimizer.__lookup_classes__)


__all__ = ["Component", "GraphFunction"] + \
          list(set(map(lambda x: x.__name__, Component.__lookup_classes__.values())))

