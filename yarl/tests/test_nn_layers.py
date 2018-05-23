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

from yarl.components.layers import DenseLayer, Conv2DLayer, ConcatLayer
from yarl.spaces import Continuous
from yarl.tests import ComponentTest

import numpy as np


class TestNNLayer(unittest.TestCase):

    def test_dense(self):
        # Space must contain batch dimension (otherwise, NNlayer will complain).
        space = Continuous(shape=(2,), add_batch_rank=True)

        # The Component to test.
        # - fixed 1.0 weights, no biases
        component_to_test = DenseLayer(units=2, weights_spec=1.0, biases_spec=False)
        test = ComponentTest(component=component_to_test, input_spaces=dict(input=space))

        # Batch of size=1 (can increase this to any larger number).
        input_ = np.array([[0.5, 2.0]])
        expected = np.array([[2.5, 2.5]])
        test.test(out_socket_name="output", inputs=input_, expected_outputs=expected)

    def test_conv2d(self):
        # Space must contain batch dimension (otherwise, NNlayer will complain).
        space = Continuous(shape=(2, 2, 3), add_batch_rank=True)

        component_to_test = Conv2DLayer(filters=4, kernel_size=2, strides=1, kernel_spec=0.5, biases_spec=False)
        test = ComponentTest(component=component_to_test, input_spaces=dict(input=space))

        # Batch of 2 samples.
        input_ = np.array([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],  # sample 1 (2x2x3)
                            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]],
                           [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],  # sample 2 (2x2x3)
                            [[0.7, 0.8, 0.9], [1.00, 1.10, 1.20]]]
                           ])
        expected = np.array([[[[39.0, 39.0, 39.0, 39.0]]],  # output 1 (1x1x4)
                             [[[3.9, 3.9, 3.9, 3.9]]],  # output 2 (1x1x4)
                             ])
        test.test(out_socket_name="output", inputs=input_, expected_outputs=expected)

    def test_concat(self):
        # Spaces must contain batch dimension (otherwise, NNlayer will complain).
        space0 = Continuous(shape=(2, 3), add_batch_rank=True)
        space1 = Continuous(shape=(2, 1), add_batch_rank=True)
        space2 = Continuous(shape=(2, 2), add_batch_rank=True)

        component_to_test = ConcatLayer(computation_inputs=3)
        test = ComponentTest(component=component_to_test, input_spaces=dict(input0=space0,
                                                                            input1=space1, input2=space2))

        # Batch of 2 samples to concatenate.
        inputs = dict(input0=np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[1.1, 2.1, 3.1], [4.1, 5.1, 6.1]]]),
                      input1=np.array([[[1.0], [2.0]], [[3.0], [4.0]]]),
                      input2=np.array([[[1.2, 2.2], [3.2, 4.2]], [[1.3, 2.3], [3.3, 4.3]]]))
        expected = np.array([[[1.0, 2.0, 3.0, 1.0, 1.2, 2.2],
                       [4.0, 5.0, 6.0, 2.0, 3.2, 4.2]],
                      [[1.1, 2.1, 3.1, 3.0, 1.3, 2.3],
                       [4.1, 5.1, 6.1, 4.0, 3.3, 4.3]]], dtype=np.float32)
        test.test(out_socket_name="output", inputs=inputs, expected_outputs=expected)

