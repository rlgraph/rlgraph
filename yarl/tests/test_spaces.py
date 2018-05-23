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

from yarl.spaces import *


class TestSpaces(unittest.TestCase):
    """
    Tests creation, sampling and shapes of Spaces.
    """
    def test_complex_space_sampling_and_check_via_contains(self):
        """
        Tests a complex Space on sampling and `contains` functionality.
        """
        space = Dict(
            a=dict(aa=float, ab=bool),
            b=dict(ba=float),
            c=float,
            d=IntBox(low=0, high=1),
            e=Discrete(5),
            f=Continuous(shape=(2,2)),
            g=Tuple(float, Continuous(shape=())),
            add_batch_rank=True
        )

        samples = space.sample(size=100, horizontal=True)
        for i in range(len(samples)):
            self.assertTrue(space.contains(samples[i]))

    def test_container_space_flattening_with_mapping(self):
        space = Tuple(
            Dict(
                a=bool,
                b=Discrete(4),
                c=Dict(
                    d=Continuous(shape=())
                )
            ),
            Bool(),
            Discrete(),
            Continuous(shape=(3, 2)),
            Tuple(
                Bool(), Bool()
            )
        )

        def mapping_func(key, primitive_space):
            # Just map a primitive Space to its flat_dim property.
            return primitive_space.flat_dim

        result = ""
        flat_space_and_mapped = space.flatten(mapping=mapping_func)
        for k, v in flat_space_and_mapped.items():
            result += "{}:{},".format(k, v)

        tuple_txt = [FLAT_TUPLE_OPEN, FLAT_TUPLE_CLOSE] * 10
        expected = "/{}0{}/a:1,/{}0{}/b:4,/{}0{}/c/d:1,/{}1{}:1,/{}2{}:2,/{}3{}:6,/{}4{}/{}0{}:1,/{}4{}/{}1{}:1,".\
            format(*tuple_txt)

        self.assertTrue(result == expected)


