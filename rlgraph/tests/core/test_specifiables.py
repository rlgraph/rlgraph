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

import numpy as np
import unittest

from rlgraph.spaces import *
from rlgraph.tests.test_util import recursive_assert_almost_equal


class TestSpecifiables(unittest.TestCase):
    """
    Tests creation of Specifiable objects via from_spec.
    """
    def test_specifiable_on_spaces(self):
        """
        Tests complex Container Spaces for being constructable from_spec.
        """
        np.random.seed(10)

        space = Dict.from_spec(
            dict(
                a=Tuple(FloatBox(shape=(1, 1, 2))),
                b=float,
                c=dict(type=float, shape=(2,))
            ), add_batch_rank=True
        )
        recursive_assert_almost_equal(
            space.sample(),
            dict(
                a=(np.array([[[0.77132064, 0.02075195]]]),),
                b=0.6336482349262754,
                c=np.array([0.74880388, 0.49850701])
            )
        )

        space = Space.from_spec(dict(type="tuple", _args=[
            Dict(
                a=bool,
                b=IntBox(4),
                c=Dict(
                    d=FloatBox(shape=())
                )
            ),
            BoolBox(),
            FloatBox(shape=(3, 2)),
            Tuple(
                bool, BoolBox()
            )]
        ))
        recursive_assert_almost_equal(
            space.sample(),
            (
                dict(
                    a=False,
                    b=0,
                    c=dict(d=0.709208009843012)
                ),
                True,
                np.array(
                    [[0.16911084, 0.08833981], [0.68535982, 0.95339335], [0.00394827, 0.51219226]], dtype=np.float32
                ),
                (True, False)
            )
        )

        space = Dict.from_spec(dict(
            a=Tuple(float, FloatBox(shape=(1, 2, 2))),
            b=FloatBox(shape=(2, 2, 2, 2)),
            c=dict(type=float, shape=(2,))
        ))
        self.assertEqual(space.rank, ((0, 3), 4, 1))
        self.assertEqual(space.shape, (((), (1, 2, 2)), (2, 2, 2, 2), (2,)))
        self.assertEqual(space.get_shape(with_batch_rank=True), (((), (1, 2, 2)), (2, 2, 2, 2), (2,)))

        space = Dict(
            a=Tuple(int, IntBox(2), FloatBox(shape=(4, 2))),
            b=FloatBox(shape=(2, 2)),
            c=dict(type=float, shape=(4,)),
            add_batch_rank=True,
            add_time_rank=True
        )
        self.assertEqual(space.rank, ((0, 0, 2), 2, 1))
        self.assertEqual(space.shape, (((), (), (4, 2)), (2, 2), (4,)))
        self.assertEqual(space.get_shape(with_batch_rank=True), (((None,), (None,), (None, 4, 2)),
                                                                 (None, 2, 2), (None, 4)))
        self.assertEqual(space.get_shape(with_time_rank=True), (((None,), (None,), (None, 4, 2)),
                                                                (None, 2, 2), (None, 4)))
        self.assertEqual(space.get_shape(with_batch_rank=True, with_time_rank=True),
                         (((None, None), (None, None), (None, None, 4, 2)), (None, None, 2, 2), (None, None, 4)))
        self.assertEqual(space.get_shape(with_batch_rank=True, with_time_rank=10, time_major=True),
                         (((10, None), (10, None), (10, None, 4, 2)), (10, None, 2, 2), (10, None, 4)))
        self.assertEqual(space.get_shape(with_batch_rank=5, with_time_rank=10, time_major=False),
                         (((5, 10), (5, 10), (5, 10, 4, 2)), (5, 10, 2, 2), (5, 10, 4)))
