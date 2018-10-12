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

import unittest
import numpy as np

from rlgraph.components.layers.nn import NNLayer, DenseLayer, Conv2DLayer, ConcatLayer, MaxPool2DLayer, \
    LSTMLayer, ResidualLayer
from rlgraph.spaces import FloatBox
from rlgraph.tests import ComponentTest
from rlgraph.utils.numpy import sigmoid, relu, lstm_layer, dense_layer


class TestNNLayer(unittest.TestCase):
    """
    Tests for the different NNLayer Components. Each layer is tested separately.
    """
    def test_dummy_nn_layer(self):
        # Tests simple pass through (no activation, no layer (graph_fn) computation).
        space = FloatBox(shape=(3,), add_batch_rank=True)

        # - fixed 1.0 weights, no biases
        dummy_layer = NNLayer(activation=None)
        test = ComponentTest(component=dummy_layer, input_spaces=dict(inputs=space))

        input_ = space.sample(size=5)
        test.test(("apply", input_), expected_outputs=input_)

    def test_activation_functions(self):
        # Test single activation functions (no other custom computations in layer).
        space = FloatBox(shape=(3,), add_batch_rank=True)

        # ReLU.
        relu_layer = NNLayer(activation="relu")
        test = ComponentTest(component=relu_layer, input_spaces=dict(inputs=space))

        input_ = space.sample(size=5)
        expected = relu(input_)
        test.test(("apply", input_), expected_outputs=expected)

        # Again manually in case util numpy-relu is broken.
        input_ = np.array([[1.0, 2.0, -5.0], [-10.0, -100.1, 4.5]])
        expected = np.array([[1.0, 2.0, 0.0], [0.0, 0.0, 4.5]])
        test.test(("apply", input_), expected_outputs=expected)

        # Sigmoid.
        sigmoid_layer = NNLayer(activation="sigmoid")
        test = ComponentTest(component=sigmoid_layer, input_spaces=dict(inputs=space))

        input_ = space.sample(size=10)
        expected = sigmoid(input_)
        test.test(("apply", input_), expected_outputs=expected)

    def test_dense_layer(self):
        # Space must contain batch dimension (otherwise, NNLayer will complain).
        space = FloatBox(shape=(2,), add_batch_rank=True)

        # - fixed 1.0 weights, no biases
        dense_layer = DenseLayer(units=2, weights_spec=1.0, biases_spec=False)
        test = ComponentTest(component=dense_layer, input_spaces=dict(inputs=space))

        # Batch of size=1 (can increase this to any larger number).
        input_ = np.array([[0.5, 2.0]])
        expected = np.array([[2.5, 2.5]])
        test.test(("apply", input_), expected_outputs=expected)

    def test_dense_layer_with_leaky_relu_activation(self):
        input_space = FloatBox(shape=(3,), add_batch_rank=True)

        dense_layer = DenseLayer(units=4, weights_spec=2.0, biases_spec=0.5, activation="lrelu")
        test = ComponentTest(component=dense_layer, input_spaces=dict(inputs=input_space))

        # Batch of size=1 (can increase this to any larger number).
        input_ = np.array([[0.5, 2.0, 1.5], [-1.0, -2.0, -1.5]])
        expected = np.array([[8.5, 8.5, 8.5, 8.5], [-8.5*0.2, -8.5*0.2, -8.5*0.2, -8.5*0.2]],
                            dtype=np.float32)  # 0.2=leaky-relu
        test.test(("apply", input_), expected_outputs=expected)

    def test_conv2d_layer(self):
        # Space must contain batch dimension (otherwise, NNlayer will complain).
        space = FloatBox(shape=(2, 2, 3), add_batch_rank=True)  # e.g. a simple 3-color image

        conv2d_layer = Conv2DLayer(filters=4, kernel_size=2, strides=1, padding="valid",
                                   kernel_spec=0.5, biases_spec=False)
        test = ComponentTest(component=conv2d_layer, input_spaces=dict(inputs=space))

        # Batch of 2 samples.
        input_ = np.array([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],  # sample 1 (2x2x3)
                            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]],
                           [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],  # sample 2 (2x2x3)
                            [[0.7, 0.8, 0.9], [1.00, 1.10, 1.20]]]
                           ])
        expected = np.array([[[[39.0, 39.0, 39.0, 39.0]]],  # output 1 (1x1x4)
                             [[[3.9, 3.9, 3.9, 3.9]]],  # output 2 (1x1x4)
                             ])
        test.test(("apply", input_), expected_outputs=expected)

    def test_maxpool2d_layer(self):
        space = FloatBox(shape=(2, 2, 3), add_batch_rank=True)  # e.g. a simple 3-color image

        # NOTE: Strides shouldn't matter.
        maxpool2d_layer = MaxPool2DLayer(pool_size=2, strides=2, padding="valid")
        test = ComponentTest(component=maxpool2d_layer, input_spaces=dict(inputs=space))

        # Batch of 2 sample.
        input_ = space.sample(2)
        item0_ch0 = max(input_[0][0][0][0], input_[0][0][1][0], input_[0][1][0][0], input_[0][1][1][0])
        item0_ch1 = max(input_[0][0][0][1], input_[0][0][1][1], input_[0][1][0][1], input_[0][1][1][1])
        item0_ch2 = max(input_[0][0][0][2], input_[0][0][1][2], input_[0][1][0][2], input_[0][1][1][2])
        item1_ch0 = max(input_[1][0][0][0], input_[1][0][1][0], input_[1][1][0][0], input_[1][1][1][0])
        item1_ch1 = max(input_[1][0][0][1], input_[1][0][1][1], input_[1][1][0][1], input_[1][1][1][1])
        item1_ch2 = max(input_[1][0][0][2], input_[1][0][1][2], input_[1][1][0][2], input_[1][1][1][2])
        expected = np.array([[[[item0_ch0, item0_ch1, item0_ch2]]], [[[item1_ch0, item1_ch1, item1_ch2]]]])
        test.test(("apply", input_), expected_outputs=expected)

    def test_concat_layer(self):
        # Spaces must contain batch dimension (otherwise, NNlayer will complain).
        space0 = FloatBox(shape=(2, 3), add_batch_rank=True)
        space1 = FloatBox(shape=(2, 1), add_batch_rank=True)
        space2 = FloatBox(shape=(2, 2), add_batch_rank=True)

        concat_layer = ConcatLayer()
        test = ComponentTest(component=concat_layer, input_spaces=dict(inputs=[space0, space1, space2]))

        # Batch of 2 samples to concatenate.
        inputs = (
            np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[1.1, 2.1, 3.1], [4.1, 5.1, 6.1]]]),
            np.array([[[1.0], [2.0]], [[3.0], [4.0]]]),
            np.array([[[1.2, 2.2], [3.2, 4.2]], [[1.3, 2.3], [3.3, 4.3]]])
        )
        expected = np.array([[[1.0, 2.0, 3.0, 1.0, 1.2, 2.2],
                              [4.0, 5.0, 6.0, 2.0, 3.2, 4.2]],
                             [[1.1, 2.1, 3.1, 3.0, 1.3, 2.3],
                              [4.1, 5.1, 6.1, 4.0, 3.3, 4.3]]], dtype=np.float32)
        test.test(("apply", inputs), expected_outputs=expected)

    def test_residual_layer(self):
        # Input space to residual layer (with 2-repeat [simple Conv2D layer]-residual-unit).
        input_space = FloatBox(shape=(2, 2, 3), add_batch_rank=True)

        residual_unit = Conv2DLayer(filters=3, kernel_size=1, strides=1, padding="same",
                                    kernel_spec=0.5, biases_spec=1.0)
        residual_layer = ResidualLayer(residual_unit=residual_unit, repeats=2)
        test = ComponentTest(component=residual_layer, input_spaces=dict(inputs=input_space))

        # Batch of 2 samples.
        inputs = np.array(
            [
                [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.1, 1.2, 1.3]]],
                [[[1.1, 1.2, 1.3], [2.4, 2.5, 2.6]], [[-0.7, -0.8, -0.9], [3.1, 3.2, 3.3]]]
            ]
        )

        """
        Calculation:
        1st_conv2d = sum-over-last-axis(input) * 0.5 + 1.0 -> tile last axis 3x
        2nd_conv2d = sum-over-last-axis(2nd_conv2d) * 0.5 + 1.0 -> tile last axis 3x
        output: 2nd_conv2d + input
        """
        conv2d_1 = np.tile(np.sum(inputs, axis=3, keepdims=True) * 0.5 + 1.0, (1, 1, 1, 3))
        conv2d_2 = np.tile(np.sum(conv2d_1, axis=3, keepdims=True) * 0.5 + 1.0, (1, 1, 1, 3))
        expected = conv2d_2 + inputs
        test.test(("apply", inputs), expected_outputs=expected, decimals=5)

    def test_lstm_layer(self):
        # 0th rank=batch-rank; 1st rank=time/sequence-rank; 2nd-nth rank=data.
        batch_size = 3
        sequence_length = 2
        input_space = FloatBox(shape=(3,), add_batch_rank=True, add_time_rank=True)

        lstm_layer_component = LSTMLayer(units=5)
        test = ComponentTest(component=lstm_layer_component, input_spaces=dict(inputs=input_space))

        # Batch of n samples.
        inputs = np.ones(shape=(batch_size, sequence_length, 3))

        # First matmul the inputs times the LSTM matrix:
        var_values = test.read_variable_values(lstm_layer_component.variables)
        lstm_matrix = var_values["lstm-layer/lstm-cell/kernel"]
        lstm_biases = var_values["lstm-layer/lstm-cell/bias"]

        expected_outputs, expected_internal_states = lstm_layer(inputs, lstm_matrix, lstm_biases, time_major=False)

        expected = dict(output=expected_outputs, last_internal_states=expected_internal_states)
        test.test(("apply", inputs), expected_outputs=expected)
