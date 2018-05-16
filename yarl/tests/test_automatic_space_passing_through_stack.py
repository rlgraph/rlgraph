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

from yarl.components.layers import GrayScale, Scale, StackComponent, Sequence
from yarl.spaces import *
from yarl.tests import ComponentTest

import numpy as np


class TestAutomaticSpacePassingThroughStack(unittest.TestCase):

    def test_two_preprocessors(self):
        # some crazy Space
        space = Dict.from_spec(dict(
            a=Continuous(shape=(1, 2)),
            b=Continuous(shape=(2, 2, 2)),
            c=Tuple(Continuous(shape=(2,)), Dict(ca=Continuous(shape=(3, 3, 2))))
        )
        )

        # Construct the Component to test (simple Stack).
        scale = Scale(scaling_factor=2)
        gray = GrayScale(weights=(0.5, 0.5), keep_rank=False)
        component_to_test = StackComponent(scale, gray)

        test = ComponentTest(component=component_to_test, input_spaces=dict(input=space))

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

        test.test(out_socket_name="output", inputs=input_, expected_outputs=expected)
        #self.assertTrue(result)

    def test_sequence_preprocessors(self):
        # some simple Space (float only)
        space = Continuous(shape=(1,), add_batch_rank=True)

        # Construct the Component to test (simple Stack).
        component_to_test = Sequence(seq_length=3, add_rank=True)

        test = ComponentTest(component=component_to_test, input_spaces=dict(input=space))

        result = test.test(out_socket_name="output", inputs=np.array([[0.1]]),
                           expected_outputs=np.array([[[0.1], [0.1], [0.1]]]))
        self.assertTrue(result)
        result = test.test(out_socket_name="output", inputs=np.array([0.2]),
                           expected_outputs=np.array([[0.2], [0.1], [0.1]]))
        self.assertTrue(result)
        result = test.test(out_socket_name="output", inputs=np.array([0.3]),
                           expected_outputs=np.array([[0.3], [0.2], [0.1]]))
        self.assertTrue(result)
        result = test.test(out_socket_name="output", inputs=np.array([0.4]),
                           expected_outputs=np.array([[0.4], [0.3], [0.2]]))
        self.assertTrue(result)
        result = test.test(out_socket_name="output", inputs=np.array([0.5]),
                           expected_outputs=np.array([[0.5], [0.4], [0.3]]))
        self.assertTrue(result)

