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

# Core.
from rlgraph.components.component import Component
# Component child-classes.
from rlgraph.components.distributions import *
from rlgraph.components.explorations import Exploration, EpsilonExploration
from rlgraph.components.layers import *
from rlgraph.components.loss_functions import *
from rlgraph.components.memories import *
from rlgraph.components.neural_networks import *
from rlgraph.components.optimizers import *
from rlgraph.components.common import *

from rlgraph.utils.util import default_dict
Component.__lookup_classes__ = dict()

# Add all specific sub-classes to this one.
default_dict(Component.__lookup_classes__, Distribution.__lookup_classes__)
default_dict(Component.__lookup_classes__, Layer.__lookup_classes__)
default_dict(Component.__lookup_classes__, Stack.__lookup_classes__)
default_dict(Component.__lookup_classes__, LossFunction.__lookup_classes__)
default_dict(Component.__lookup_classes__, Memory.__lookup_classes__)
default_dict(Component.__lookup_classes__, NeuralNetwork.__lookup_classes__)
default_dict(Component.__lookup_classes__, Optimizer.__lookup_classes__)


__all__ = ["Component"] + \
          list(set(map(lambda x: x.__name__, Component.__lookup_classes__.values())))

