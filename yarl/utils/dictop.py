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


class dictop(dict):
    """
    A hashable dict that's used in YARL only to make dict-type ops hashable so we can store them in sets and use them as
    lookup keys in other dicts.
    Dict() Spaces produce dictops when methods like get_tensor_variables are called on them.
    """
    def __hash__(self):
        """
        Hash based on sequence of sorted items (keys are all strings, values are always other dictops, tuples or
        primitive ops).
        """
        return hash(tuple(sorted(self.items())))

