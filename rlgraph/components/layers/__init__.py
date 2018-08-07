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

from rlgraph.utils.util import default_dict

# Basics.
from rlgraph.components.layers.layer import Layer
# Preprocessing Layers.
from rlgraph.components.layers.preprocessing import *
# NN-Layers.
from rlgraph.components.layers.nn import *
# String Layers.
from rlgraph.components.layers.strings import *

# The Layers (Layers are also Stacks).
Layer.__lookup_classes__ = dict(
    nnlayer=NNLayer,
    preprocesslayer=PreprocessLayer
)
# Add all specific Layer sub-classes to this one.
default_dict(Layer.__lookup_classes__, NNLayer.__lookup_classes__)
default_dict(Layer.__lookup_classes__, PreprocessLayer.__lookup_classes__)
default_dict(Layer.__lookup_classes__, StringLayer.__lookup_classes__)


__all__ = ["Layer"] + \
          list(set(map(lambda x: x.__name__, Layer.__lookup_classes__.values())))
