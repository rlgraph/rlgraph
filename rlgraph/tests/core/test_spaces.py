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

from six.moves import xrange as range_
import unittest

from rlgraph.spaces import *
from rlgraph.utils.ops import FLAT_TUPLE_CLOSE, FLAT_TUPLE_OPEN


class TestSpaces(unittest.TestCase):
    """
    Tests creation, sampling and shapes of Spaces.
    """
    def test_box_spaces(self):
        """
        Tests all BoxSpaces via sample/contains loop. With and without batch-rank,
        different batch sizes, and different los/high combinations (including no bounds).
        """
        for class_ in [FloatBox, IntBox, BoolBox, TextBox]:
            for add_batch_rank in [False, True]:
                # TODO: Test time-rank more thoroughly.
                for add_time_rank in [False, True]:
                    if class_ != BoolBox and class_ != TextBox:
                        for low, high in [(None, None), (-1.0, 10.0), ((1.0, 2.0), (3.0, 4.0)),
                                          (((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)), ((7.0, 8.0, 9.0), (10.0, 11.0, 12.0)))]:
                            space = class_(low=low, high=high, add_batch_rank=add_batch_rank,
                                           add_time_rank=add_time_rank)
                            if add_batch_rank is False:
                                sample = space.sample()
                                self.assertTrue(space.contains(sample))
                            else:
                                for batch_size in range_(1, 4):
                                    samples = space.sample(size=batch_size)
                                    for s in samples:
                                        self.assertTrue(space.contains(s))
                            # TODO: test zero() method perperly for all cases
                            #all_0s = space.zeros()
                            #self.assertTrue(all(v == 0 for v in all_0s))
                    else:
                        space = class_(add_batch_rank=add_batch_rank, add_time_rank=add_time_rank)
                        if add_batch_rank is False:
                            sample = space.sample()
                            self.assertTrue(space.contains(sample))
                        else:
                            for batch_size in range_(1, 4):
                                samples = space.sample(size=batch_size)
                                for s in samples:
                                    self.assertTrue(space.contains(s))

    def test_complex_space_sampling_and_check_via_contains(self):
        """
        Tests a complex Space on sampling and `contains` functionality.
        """
        space = Dict(
            a=dict(aa=float, ab=bool),
            b=dict(ba=float),
            c=float,
            d=IntBox(low=0, high=1),
            e=IntBox(5),
            f=FloatBox(shape=(2,2)),
            g=Tuple(float, FloatBox(shape=())),
            add_batch_rank=True
        )

        samples = space.sample(size=100, horizontal=True)
        for i in range_(len(samples)):
            self.assertTrue(space.contains(samples[i]))

    def test_container_space_flattening_with_mapping(self):
        space = Tuple(
            Dict(
                a=bool,
                b=IntBox(4),
                c=Dict(
                    d=FloatBox(shape=())
                )
            ),
            BoolBox(),
            IntBox(2),
            FloatBox(shape=(3, 2)),
            Tuple(
                BoolBox(), BoolBox()
            )
        )

        def mapping_func(key, primitive_space):
            # Just map a primitive Space to its flat_dim property.
            return primitive_space.flat_dim

        result = ""
        flat_space_and_mapped = space.flatten(mapping=mapping_func)
        for key, value in flat_space_and_mapped.items():
            result += "{}:{},".format(key, value)

        tuple_txt = [FLAT_TUPLE_OPEN, FLAT_TUPLE_CLOSE] * 10
        expected = "{}0{}/a:1,{}0{}/b:1,{}0{}/c/d:1,{}1{}:1,{}2{}:1,{}3{}:6,{}4{}/{}0{}:1,{}4{}/{}1{}:1,".\
            format(*tuple_txt)

        self.assertTrue(result == expected)


