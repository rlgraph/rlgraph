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

import cv2
import numpy as np
from six.moves import xrange as range_
import unittest

from rlgraph.components.layers import GrayScale, Flatten, Multiply, PreprocessorStack, Sequence, Clip, \
    ImageBinary, ImageResize
from rlgraph.spaces import *
from rlgraph.tests import ComponentTest


class TestPreprocessors(unittest.TestCase):

    def test_clip(self):
        clip = Clip(min=0.0, max=1.0)
        # Grayscale image of 2x2 size.
        test = ComponentTest(component=clip, input_spaces=dict(apply=FloatBox(shape=(2, 2), add_batch_rank=True)))

        test.test("reset")
        # Batch=3
        input_images = np.array([
            [[125.6, 10.3], [-45, 5.234]],
            [[-10.0, 1.0004], [0.0, -0.0003]],
            [[0.0005, 0.00000009], [90.0, 10000901.347]]
        ])
        expected = np.array([
            [[1.0, 1.0], [0.0, 1.0]],
            [[0.0, 1.0], [0.0, 0.0]],
            [[0.0005, 0.00000009], [1.0, 1.0]]
        ])
        test.test(("apply", input_images), expected_outputs=expected)

    def test_black_and_white(self):
        binary = ImageBinary()
        # Color image of 2x2x3 size.
        test = ComponentTest(component=binary, input_spaces=dict(apply=FloatBox(shape=(2, 2, 3), add_batch_rank=True)))

        test.test("reset")
        # Batch=2
        input_images = np.array([
            [[[0, 1, 0], [10, 9, 5]], [[0, 0, 0], [0, 0, 1]]],
            [[[255, 255, 255], [0, 0, 0]], [[0, 0, 0], [255, 43, 0]]]
        ])
        expected = np.array([
            [[1, 1], [0, 1]],
            [[1, 0], [0, 1]]
        ])
        test.test(("apply", input_images), expected_outputs=expected)

    def test_simple_preprocessor_stack_with_one_preprocess_layer(self):
        stack = PreprocessorStack(dict(type="multiply", factor=0.5))

        test = ComponentTest(component=stack, input_spaces=dict(preprocess=float))

        test.test("reset")
        test.test(("preprocess", 2.0), expected_outputs=1.0)

    def test_preprocessor_from_list_spec(self):
        space = FloatBox(shape=(2,))
        stack = PreprocessorStack.from_spec([
            dict(type="grayscale", keep_rank=False, weights=(0.5, 0.5)),
            dict(type="divide", divisor=2),
        ])
        test = ComponentTest(component=stack, input_spaces=dict(preprocess=space))

        # Run the test.
        input_ = np.array([3.0, 5.0])
        expected = np.array(2.0)
        test.test("reset")
        test.test(("preprocess", input_), expected_outputs=expected)

    def test_split_inputs_on_grayscale(self):
        # last rank is always the color rank (its dim must match len(grayscale-weights))
        space = Dict.from_spec(dict(
            a=Tuple(FloatBox(shape=(1, 1, 2)), FloatBox(shape=(1, 2, 2))),
            b=FloatBox(shape=(2, 2, 2, 2)),
            c=dict(type=float, shape=(2,))  # single scalar pixel
        ))

        test = ComponentTest(component=GrayScale(weights=(0.5, 0.5), keep_rank=False),
                             input_spaces=dict(apply=space))

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
        test.test("reset")
        test.test(("apply", input_), expected_outputs=expected)

    def test_split_graph_on_flatten(self):
        space = Dict.from_spec(
            dict(
                a=Tuple(FloatBox(shape=(1, 1, 2)), FloatBox(shape=(1, 2, 2))),
                b=FloatBox(shape=(2, 2, 3)),
                c=dict(type=float, shape=(2,)),
                d=IntBox(3)
            ),
            add_batch_rank=True
        )

        test = ComponentTest(component=Flatten(), input_spaces=dict(apply=space))

        input_ = dict(
            a=(
                np.array([[[[3.0, 5.0]]], [[[1.0, 5.2]]]]), np.array([[[[3.1, 3.2], [3.3, 3.4]]],
                                                                      [[[3.5, 3.6], [3.7, 3.8]]]])
            ),
            b=np.array([[[[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]], [[0.07, 0.08, 0.09], [0.10, 0.11, 0.12]]],
                        [[[0.13, 0.14, 0.15], [0.16, 0.17, 0.18]], [[0.19, 0.20, 0.21], [0.22, 0.23, 0.24]]]]),
            c=np.array([[0.1, 0.2], [0.3, 0.4]]),
            d=np.array([2, 0])
        )
        expected = dict(
            a=(
                np.array([[3.0, 5.0], [1.0, 5.2]], dtype=np.float32), np.array([[3.1, 3.2, 3.3, 3.4], [3.5, 3.6, 3.7, 3.8]], dtype=np.float32)
            ),
            b=np.array([[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12],
                        [0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24]]
            ),
            c=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
            d=np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])  # category (one-hot) flatten
        )
        test.test("reset")
        test.test(("apply", input_), expected_outputs=expected)

    def test_two_preprocessors_in_a_preprocessor_stack(self):
        space = Dict(
            a=FloatBox(shape=(1, 2)),
            b=FloatBox(shape=(2, 2, 2)),
            c=Tuple(FloatBox(shape=(2,)), Dict(ca=FloatBox(shape=(3, 3, 2))))
        )

        # Construct the Component to test (PreprocessorStack).
        scale = Multiply(factor=2)
        gray = GrayScale(weights=(0.5, 0.5), keep_rank=False)
        test = ComponentTest(component=PreprocessorStack(scale, gray), input_spaces=dict(preprocess=space))

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
        test.test("reset")
        test.test(("preprocess", input_), expected_outputs=expected)

    def test_sequence_preprocessor(self):
        space = FloatBox(shape=(1,), add_batch_rank=True)
        component_to_test = Sequence(length=3, add_rank=True)
        test = ComponentTest(component=component_to_test, input_spaces=dict(apply=space))

        vars = component_to_test.get_variables("index", "buffer", global_scope=False)
        index, buffer = vars["index"], vars["buffer"]

        for _ in range_(3):
            test.test("reset")
            index_value, buffer_value = test.get_variable_values(index, buffer)
            self.assertEqual(index_value, -1)
            test.test(("apply", np.array([[0.1]])),
                      expected_outputs=np.array([[[0.1, 0.1, 0.1]]]))
            index_value, buffer_value = test.get_variable_values(index, buffer)
            self.assertEqual(index_value, 0)
            test.test(("apply", np.array([[0.2]])),
                      expected_outputs=np.array([[[0.1, 0.1, 0.2]]]))
            index_value, buffer_value = test.get_variable_values(index, buffer)
            self.assertEqual(index_value, 1)
            test.test(("apply", np.array([[0.3]])),
                      expected_outputs=np.array([[[0.1, 0.2, 0.3]]]))
            index_value, buffer_value = test.get_variable_values(index, buffer)
            self.assertEqual(index_value, 2)
            test.test(("apply", np.array([[0.4]])),
                      expected_outputs=np.array([[[0.2, 0.3, 0.4]]]))
            index_value, buffer_value = test.get_variable_values(index, buffer)
            self.assertEqual(index_value, 0)
            test.test(("apply", np.array([[0.5]])),
                      expected_outputs=np.array([[[0.3, 0.4, 0.5]]]))
            index_value, buffer_value = test.get_variable_values(index, buffer)
            self.assertEqual(index_value, 1)

            ## TODO: Insert more than one sample.
            #test.test(("apply", np.array([[0.6], [0.7]])),
            #          expected_outputs=np.array([[[0.5, 0.6, 0.7]]]))
            #index_value, buffer_value = test.get_variable_values(index, buffer)
            #self.assertEqual(index_value, 0)

    def test_sequence_preprocessor_with_container_space(self):
        # Test with no batch rank.
        space = Tuple(FloatBox(shape=(1,)), FloatBox(shape=(2, 2)), add_batch_rank=False)

        component_to_test = Sequence(length=4, add_rank=False)
        test = ComponentTest(component=component_to_test, input_spaces=dict(apply=space))

        for i in range_(3):
            test.test("reset")
            test.test(("apply", np.array([np.array([0.5]), np.array([[0.6, 0.7], [0.8, 0.9]])])),
                      expected_outputs=(np.array([0.5, 0.5, 0.5, 0.5]), np.array([[0.6, 0.7] * 4,
                                                                                  [0.8, 0.9] * 4])))
            test.test(("apply", np.array([np.array([0.6]), np.array([[1.1, 1.1], [1.1, 1.1]])])),
                      expected_outputs=(np.array([0.5, 0.5, 0.5, 0.6]), np.array([[0.6, 0.7, 0.6, 0.7,
                                                                                   0.6, 0.7, 1.1, 1.1],
                                                                                  [0.8, 0.9, 0.8, 0.9,
                                                                                   0.8, 0.9, 1.1, 1.1]])))
            test.test(("apply", np.array([np.array([0.7]), np.array([[2.0, 2.1], [2.2, 2.3]])])),
                      expected_outputs=(np.array([0.5, 0.5, 0.6, 0.7]), np.array([[0.6, 0.7, 0.6, 0.7,
                                                                                   1.1, 1.1, 2.0, 2.1],
                                                                                  [0.8, 0.9, 0.8, 0.9,
                                                                                   1.1, 1.1, 2.2, 2.3]])))

    def test_image_resize(self):
        image_resize = ImageResize(width=4, height=4)
        # Some image of 16x16x3 size.
        test = ComponentTest(
            component=image_resize, input_spaces=dict(apply=FloatBox(shape=(16, 16, 3), add_batch_rank=False))
        )

        test.test("reset")

        input_image = cv2.imread("images/16x16x3_image.bmp")
        expected = cv2.imread("images/4x4x3_image_resized.bmp")
        test.test(("apply", input_image), expected_outputs=expected)
