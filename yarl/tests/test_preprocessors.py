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

from yarl.components.layers import GrayScale, Flatten, Scale, StackComponent, Sequence
from yarl.spaces import *
from yarl.tests import ComponentTest

import numpy as np


class TestPreprocessors(unittest.TestCase):

    def test_split_computation_on_grayscale(self):
        # last rank is always the color rank (its dim must match len(grayscale-weights))
        space = Dict.from_spec(dict(
            a=Tuple(Continuous(shape=(1, 1, 2)), Continuous(shape=(1, 2, 2))),
            b=Continuous(shape=(2, 2, 2, 2)),
            c=dict(type=float, shape=(2,))  # single scalar pixel
        ))

        test = ComponentTest(component=GrayScale(weights=(0.5, 0.5), keep_rank=False),
                             input_spaces=dict(input=space))

        # Run the test.
        input_ = dict(
            a=(
                np.array([[[3.0, 5.0]]]), np.array([[[3.0, 5.0], [1.0, 5.0]]])
            ),
            b=np.array([[[[2.0, 4.0], [2.0, 4.0]],
                         [[2.0, 4.0], [2.0, 4.0]]],
                        [[[2.0, 4.0], [2.0, 4.0]],
                         [[2.0, 4.0], [2.0, 4.0]]]]
                       ),
            c=np.array([0.6, 0.8])
        )
        expected = dict(
            a=(
                np.array([[4.0]]), np.array([[4.0, 3.0]])
            ),
            b=np.array([[[3.0, 3.0], [3.0, 3.0]], [[3.0, 3.0], [3.0, 3.0]]]),
            c=0.7
        )
        test.test(out_socket_name="reset")
        test.test(out_socket_name="output", inputs=input_, expected_outputs=expected)

    def test_split_computation_on_flatten(self):
        space = Dict.from_spec(dict(
            a=Tuple(Continuous(shape=(1, 1, 2)), Continuous(shape=(1, 2, 2))),
            b=Continuous(shape=(2, 2, 3)),
            c=dict(type=float, shape=(2,)),
            add_batch_rank=True
        ))

        test = ComponentTest(component=Flatten(), input_spaces=dict(input=space))

        input_ = dict(
            a=(
                np.array([[[[3.0, 5.0]]], [[[1.0, 5.2]]]]), np.array([[[[3.1, 3.2], [3.3, 3.4]]],
                                                                      [[[3.5, 3.6], [3.7, 3.8]]]])
            ),
            b=np.array([[[[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]], [[0.07, 0.08, 0.09], [0.10, 0.11, 0.12]]],
                        [[[0.13, 0.14, 0.15], [0.16, 0.17, 0.18]], [[0.19, 0.20, 0.21], [0.22, 0.23, 0.24]]]]),
            c=np.array([[0.1, 0.2], [0.3, 0.4]])
        )
        expected = dict(
            a=(
                np.array([[3.0, 5.0], [1.0, 5.2]], dtype=np.float32), np.array([[3.1, 3.2, 3.3, 3.4], [3.5, 3.6, 3.7, 3.8]], dtype=np.float32)
            ),
            b=np.array([[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12],
                        [0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24]]
            ),
            c=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        )
        test.test(out_socket_name="reset")
        test.test(out_socket_name="output", inputs=input_, expected_outputs=expected)

    def test_two_preprocessors(self):
        space = Dict(
            a=Continuous(shape=(1, 2)),
            b=Continuous(shape=(2, 2, 2)),
            c=Tuple(Continuous(shape=(2,)), Dict(ca=Continuous(shape=(3, 3, 2))))
        )

        # Construct the Component to test (simple Stack).
        scale = Scale(scaling_factor=2)
        gray = GrayScale(weights=(0.5, 0.5), keep_rank=False)
        test = ComponentTest(component=StackComponent(scale, gray), input_spaces=dict(input=space))

        input_ = dict(
            a=np.array([[3.0, 5.0]]),
            b=np.array([[[2.0, 4.0], [2.0, 4.0]], [[2.0, 4.0], [2.0, 4.0]]]),
            c=(np.array([10.0, 20.0]), dict(ca=np.array([[[1.0, 2.0],[1.0, 2.0],[1.0, 2.0]],
                                                         [[1.0, 2.0],[1.0, 2.0],[1.0, 2.0]],
                                                         [[1.0, 2.0],[1.0, 2.0],[1.0, 2.0]]])))
        )
        expected = dict(
            a=np.array([8.0]),
            b=np.array([[6.0, 6.0], [6.0, 6.0]]),
            c=(30.0, dict(ca=np.array([[3.0, 3.0, 3.0],
                                       [3.0, 3.0, 3.0],
                                       [3.0, 3.0, 3.0]])))
        )
        test.test(out_socket_name="reset")
        test.test(out_socket_name="output", inputs=input_, expected_outputs=expected)

    def test_sequence_preprocessor(self):
        space = Tuple(Continuous(shape=(1,)), Continuous(shape=(2, 2)), add_batch_rank=True)

        # Construct the Component to test (simple Stack).
        component_to_test = Sequence(seq_length=3, add_rank=True)

        test = ComponentTest(component=component_to_test, input_spaces=dict(input=space))

        for i in range(3):
            test.test(out_socket_name="reset")
            test.test(out_socket_name="output", inputs=(np.array([[0.1]]), np.array([[[0.2, 0.3], [0.4, 0.5]]])),
                      expected_outputs=(np.array([[[0.1], [0.1], [0.1]]]), np.array([[
                          [[0.2, 0.3], [0.4, 0.5]],
                          [[0.2, 0.3], [0.4, 0.5]],
                          [[0.2, 0.3], [0.4, 0.5]]
                      ]])))
            """
            test.test(out_socket_name="output", inputs=np.array([0.2]),
                      expected_outputs=np.array([[0.2], [0.1], [0.1]]))
            test.test(out_socket_name="output", inputs=np.array([0.3]),
                      expected_outputs=np.array([[0.3], [0.2], [0.1]]))
            test.test(out_socket_name="output", inputs=np.array([0.4]),
                      expected_outputs=np.array([[0.4], [0.3], [0.2]]))
            test.test(out_socket_name="output", inputs=np.array([0.5]),
                      expected_outputs=np.array([[0.5], [0.4], [0.3]]))
            """

