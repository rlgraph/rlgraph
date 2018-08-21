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

from copy import deepcopy
import numpy as np
from six.moves import xrange as range_
import unittest

from rlgraph.components.neural_networks.dict_preprocessor_stack import DictPreprocessorStack
from rlgraph.spaces import *
from rlgraph.tests import ComponentTest, recursive_assert_almost_equal
from rlgraph.tests.test_util import config_from_path


class TestDictPreprocessorStacks(unittest.TestCase):
    """
    Tests dict preprocessor stacks.
    """
    def test_dict_preprocessor_stack(self):
        """
        Tests if Python and TensorFlow backend return the same output
        for a standard DQN-style preprocessing stack.
        """
        input_space = Dict(
            a=FloatBox(shape=(2, 3)),
            b=IntBox(3),
            c=FloatBox(shape=(4, 5, 6)),
            add_batch_rank=True
        )
        preprocessors = dict(
            a=[dict(type="divide", divisor=2), dict(type="multiply", factor=4)],
            c=[dict(type="reshape", flatten=True)]
        )

        dict_preprocessor_stack = DictPreprocessorStack(preprocessors)

        test = ComponentTest(component=dict_preprocessor_stack, input_spaces=dict(inputs=input_space))

        # Run the test.
        batch_size = 5
        inputs = input_space.sample(batch_size)
        expected = dict(a=inputs["a"] * 2, b=inputs["b"], c=np.reshape(inputs["c"], newshape=(batch_size, 120,)))
        test.test("reset")
        test.test(("preprocess", inputs), expected_outputs=expected)
