# Copyright 2018/2019 ducandu GmbH. All Rights Reserved.
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
from rlgraph.components.neural_networks.multi_input_stream_neural_network import MultiInputStreamNeuralNetwork
from rlgraph.spaces import FloatBox, IntBox, Dict, Tuple
from rlgraph.tests.component_test import ComponentTest
from rlgraph.utils.numpy import dense_layer, one_hot


class TestMultiInputStreamNeuralNetwork(unittest.TestCase):
    """
    Tests for the VariationalAutoEncoder class.
    """
    def test_multi_input_stream_neural_network_with_tuple(self):
        # Space must contain batch dimension (otherwise, NNLayer will complain).
        input_space = Tuple(
            IntBox(3, shape=()),
            FloatBox(shape=(8,)),
            IntBox(4, shape=()),
            add_batch_rank=True
        )

        multi_input_nn = MultiInputStreamNeuralNetwork(
            input_network_specs=(
                [{"type": "reshape", "flatten": True, "flatten_categories": True}],  # intbox -> flatten
                [{"type": "dense", "units": 2}],  # floatbox -> dense
                [{"type": "reshape", "flatten": True, "flatten_categories": True}]  # inbox -> flatten
            ),
            post_network_spec=[{"type": "dense", "units": 3}],
        )

        test = ComponentTest(component=multi_input_nn, input_spaces=dict(inputs=input_space))

        # Batch of size=n.
        nn_inputs = input_space.sample(3)

        global_scope_pre = "multi-input-stream-nn/input-stream-nn-"
        global_scope_post = "multi-input-stream-nn/post-concat-nn/dense-layer/dense/"
        # Calculate output manually.
        var_dict = test.read_variable_values()

        flat_0 = one_hot(nn_inputs[0], depth=3)
        dense_1 = dense_layer(
            nn_inputs[1], var_dict[global_scope_pre+"1/dense-layer/dense/kernel"],
            var_dict[global_scope_pre+"1/dense-layer/dense/bias"]
        )
        flat_2 = one_hot(nn_inputs[2], depth=4)
        concat_out = np.concatenate((flat_0, dense_1, flat_2), axis=-1)
        expected = dense_layer(concat_out, var_dict[global_scope_post+"kernel"], var_dict[global_scope_post+"bias"])

        test.test(("call", tuple([nn_inputs])), expected_outputs=expected)

        test.terminate()

    def test_multi_input_stream_neural_network_with_dict(self):
        # Space must contain batch dimension (otherwise, NNlayer will complain).
        input_space = Dict(
            a=FloatBox(shape=(3,)),
            b=IntBox(4, shape=()),
            add_batch_rank=True
        )

        multi_input_nn = MultiInputStreamNeuralNetwork(
            input_network_specs=dict(
                a=[],
                b=[{"type": "reshape", "flatten": True, "flatten_categories": True}]
            ),
            post_network_spec=[{"type": "dense", "units": 2}],
        )

        test = ComponentTest(component=multi_input_nn, input_spaces=dict(inputs=input_space))

        # Batch of size=n.
        nn_inputs = input_space.sample(5)

        global_scope = "multi-input-stream-nn/post-concat-nn/dense-layer/dense/"
        # Calculate output manually.
        var_dict = test.read_variable_values()

        b_flat = one_hot(nn_inputs["b"], depth=4)
        concat_out = np.concatenate((nn_inputs["a"], b_flat), axis=-1)
        expected = dense_layer(concat_out, var_dict[global_scope+"kernel"], var_dict[global_scope+"bias"])

        test.test(("call", nn_inputs), expected_outputs=expected)

        test.terminate()
