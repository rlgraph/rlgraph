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

from yarl.spaces import Dict, Tuple, Continuous, Bool, Discrete

space = Tuple(
    Dict(
        a=Bool(),  # tuple-0/a
        b=Discrete(4),  # tuple-0/b
        c=Dict(
            d=Continuous()  # tuple-0/c/d
        )
    ),
    Bool(),  # tuple-1/
    Discrete(),  # tuple-2/
    Continuous(3),  # tuple-3/
    Tuple(
        Bool()  # tuple-4/tuple-0/
    )
)


def mapping_func(primitive_space):
    return primitive_space.flat_dim


flat_space_and_mapped = space.flatten(mapping=mapping_func)

for k, v in flat_space_and_mapped.items():
    print(k+": "+str(v))
