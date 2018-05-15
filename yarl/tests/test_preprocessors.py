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

from yarl.components.layers import GrayScale
from yarl.spaces import *
from yarl.tests import ComponentTest

import numpy as np


#class TestPreprocessors(unittest.TestCase):

#    def test_split_computation_on_grayscale(self):
"""space = Dict.from_spec(dict(
    a=dict(
        a=Continuous(shape=(1, 1, 3)),  # 2D image
        b=Continuous(shape=(2, 2, 2, 3)),  # 3D image
        c=dict(
            ca=Continuous(shape=(1, 3)),  # 1D image
            cb=Continuous(shape=(4, 4, 3))  # 2D image
        )
    ),
    b=Continuous(shape=(3,)),  # 0D image (pixel)
))"""

# last rank is always the color rank (its dim must match len(grayscale-weights))
space = Dict.from_spec(dict(
    a=Tuple(Continuous(shape=(1, 1, 2)), Continuous(shape=(1, 2, 2))),
    b=Continuous(shape=(2, 2, 2, 2)),
))

# The Component to test.
component_to_test = GrayScale(weights=(0.5, 0.5))

# A ComponentTest object.
test = ComponentTest(component=component_to_test, input_spaces=dict(input=space))

# Run the test.
input_ = dict(
    a=(
        np.array([[[3.0, 5.0]]]), np.array([[[3.0, 5.0], [1.0, 5.0]]])
    ),
    b=np.array([[[[2.0, 4.0], [2.0, 4.0]], [[2.0, 4.0], [2.0, 4.0]]], [[[2.0, 4.0], [2.0, 4.0]], [[2.0, 4.0], [2.0, 4.0]]]])
)
expected = dict(
    a=(
        np.array([[4.0]]), np.array([[4.0, 3.0]])
    ),
    b=np.array([[[3.0, 3.0], [3.0, 3.0]], [[3.0, 3.0], [3.0, 3.0]]])
)

result = test.test(out_socket_name="output", inputs=input_, expected_outputs=expected)
assert result, "FAIL!"

print("PASS!")

