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
import numpy as np

from rlgraph.spaces.space import Space
from rlgraph.spaces.box_space import BoxSpace
from rlgraph.spaces.float_box import FloatBox
from rlgraph.spaces.int_box import IntBox
from rlgraph.spaces.bool_box import BoolBox
from rlgraph.spaces.containers import ContainerSpace, Dict, Tuple
from rlgraph.spaces.space_utils import sanity_check_space


Space.__lookup_classes__ = dict({
    "bool": BoolBox,
    bool: BoolBox,
    np.bool_: BoolBox,
    "int": IntBox,
    int: IntBox,
    np.int32: IntBox,
    "intbox": IntBox,
    "multidiscrete": IntBox,
    "continuous": FloatBox,
    "float": FloatBox,
    float: FloatBox,
    np.float32: FloatBox,
    np.float64: FloatBox,
    "list": Tuple,
    "tuple": Tuple,
    # "sequence" action type for nlp use cases and combinatorial optimisation.
    "sequence": Tuple,
    dict: Dict,
    "dict": Dict
})

# Default Space: A float from 0.0 to 1.0.
Space.__default_constructor__ = partial(FloatBox, 1.0)


__all__ = ["Space", "BoxSpace", "FloatBox", "IntBox", "BoolBox",
           "ContainerSpace", "Dict", "Tuple",
           "sanity_check_space"]

