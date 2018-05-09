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

from .space import Space
from .discrete import Discrete, Bool
from .continuous import Continuous
from .intbox import IntBox
from .containers import Space, Dict, Tuple


Space.__lookup_classes__ = {
    "bool": Bool,
    bool: Bool,
    "discrete": Discrete,
    "int": Discrete,
    int: Discrete,
    "intbox": IntBox,
    "multidiscrete": IntBox,
    "continuous": Continuous,
    "float": Continuous,
    float: Continuous,
    "tuple": Tuple,
    # "sequence" action type for nlp use cases and combinatorial optimisation.
    "sequence": Tuple,
    "dict": Dict
}

__all__ = ["Space", "Discrete", "Bool", "Continuous", "IntBox", "ContainerSpace", "Dict", "Tuple"]



