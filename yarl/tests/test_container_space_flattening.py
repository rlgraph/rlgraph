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

import unittest

from yarl.spaces import Dict, Tuple, Continuous, Bool, Discrete


class TestSpaces(unittest.TestCase):

    def test_container_space_flattening_with_mapping(self):
        space = Tuple(
            Dict(
                a=Bool(),
                b=Discrete(4),
                c=Dict(
                    d=Continuous()
                )
            ),
            Bool(),
            Discrete(),
            Continuous(shape=(3, 2)),
            Tuple(
                Bool()
            )
        )

        def mapping_func(key, primitive_space):
            # Just map a primitive Space to its flat_dim property.
            return primitive_space.flat_dim

        result = ""
        flat_space_and_mapped = space.flatten(mapping=mapping_func)
        for k, v in flat_space_and_mapped.items():
            result += "{}:{},".format(k, v)

        expected = "/tuple-0/a:1,/tuple-0/b:4,/tuple-0/c/d:1,/tuple-1:1,/tuple-2:2,/tuple-3:6,/tuple-4/tuple-0:1,"

        self.assertTrue(result == expected)


