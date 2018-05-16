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
    def test_complex_space(self):
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

        samples = space.sample(size=10)
        for i in range(len(samples)):
            print(space.contains(samples[i]))

