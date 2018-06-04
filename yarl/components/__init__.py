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
from .component import Component, CONNECT_INS, CONNECT_OUTS
from .socket_and_graph_fn import Socket, GraphFunction
# Component child-classes.
from .common import *
from .distributions import *
from .explorations import *
from .layers import *
from .loss_functions import *
from .memories import *
from .neural_networks import *
from .optimizers import *

Component.__lookup_classes__ = dict()
# Add all specific sub-classes to this one.
default_dict(Component.__lookup_classes__, Distribution.__lookup_classes__)
default_dict(Component.__lookup_classes__, Layer.__lookup_classes__)
default_dict(Component.__lookup_classes__, Stack.__lookup_classes__)
default_dict(Component.__lookup_classes__, LossFunction.__lookup_classes__)
default_dict(Component.__lookup_classes__, Memory.__lookup_classes__)
default_dict(Component.__lookup_classes__, NeuralNetwork.__lookup_classes__)
default_dict(Component.__lookup_classes__, Optimizer.__lookup_classes__)


__all__ = ["Component", #"Splitter", "Merger",
           "CONNECT_INS", "CONNECT_OUTS",
           "GraphFunction", "Socket"]

