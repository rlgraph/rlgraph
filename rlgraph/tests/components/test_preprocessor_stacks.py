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
from six.moves import xrange as range_
import unittest

from rlgraph.components.layers import GrayScale, Multiply
from rlgraph.components.neural_networks import PreprocessorStack
from rlgraph.spaces import *
from rlgraph.tests import ComponentTest, recursive_assert_almost_equal


class TestPreprocessorStacks(unittest.TestCase):

    def test_simple_preprocessor_stack_with_one_preprocess_layer(self):
        stack = PreprocessorStack(dict(type="multiply", factor=0.5))

        test = ComponentTest(component=stack, input_spaces=dict(preprocess=float))

        test.test("reset")
        test.test(("preprocess", 2.0), expected_outputs=1.0)

    # TODO: Make it irrelevent whether we test a python or a tf Component (API and handling should be 100% identical)
    def test_simple_python_preprocessor_stack(self):
        space = FloatBox(shape=(2,), add_batch_rank=True)
        # python PreprocessorStack
        multiply = dict(type="multiply", factor=0.5, scope="m")
        divide = dict(type="divide", divisor=0.5, scope="d")
        stack = PreprocessorStack(multiply, divide, backend="python")
        for sub_comp_scope in ["m", "d"]:
            stack.sub_components[sub_comp_scope].create_variables(input_spaces=dict(inputs=space))

        #test = ComponentTest(component=stack, input_spaces=dict(preprocess=float))

        for _ in range_(3):
            # Call fake API-method directly (ok for PreprocessorStack).
            stack.reset()
            input_ = np.asarray([[1.0], [2.0], [3.0], [4.0]])
            out = stack.preprocess(input_)
            recursive_assert_almost_equal(out, input_)

            input_ = space.sample()
            out = stack.preprocess(input_)
            recursive_assert_almost_equal(out, input_)

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

    def test_two_preprocessors_in_a_preprocessor_stack(self):
        space = Dict(
            a=FloatBox(shape=(1, 2)),
            b=FloatBox(shape=(2, 2, 2)),
            c=Tuple(FloatBox(shape=(2,)), Dict(ca=FloatBox(shape=(3, 3, 2))))
        )

        # Construct the Component to test (PreprocessorStack).
        scale = Multiply(factor=2)
        gray = GrayScale(weights=(0.5, 0.5), keep_rank=False)
        stack = PreprocessorStack(scale, gray)
        test = ComponentTest(component=stack, input_spaces=dict(preprocess=space))

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
