# Copyright 2018/2019 The RLgraph authors. All Rights Reserved.
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

from __future__ import absolute_import, division, print_function

import unittest

from rlgraph.components.layers.nn.dense_layer import DenseLayer
from rlgraph.components.neural_networks.value_function import ValueFunction
from rlgraph.spaces import FloatBox
from rlgraph.tests.component_test import ComponentTest
from rlgraph.tests.test_util import config_from_path
from rlgraph.utils.numpy import dense_layer
from rlgraph.utils.rlgraph_errors import RLGraphError


class TestValueFunctions(unittest.TestCase):
    """
    Tests for assembling from json and running different NeuralNetworks.
    """
    def test_simple_value_function_using_layers(self):
        # Space must contain batch dimension (otherwise, NNlayer will complain).
        space = FloatBox(shape=(4,), add_batch_rank=True)

        # Create a simple neural net from json.
        nn_layers = config_from_path("configs/test_simple_nn.json")
        value_function = ValueFunction(*nn_layers["layers"])

        # Do not seed, we calculate expectations manually.
        test = ComponentTest(component=value_function, input_spaces=dict(inputs=space))

        # Batch of size=3.
        input_ = space.sample(4)
        # Calculate output manually.
        var_dict = test.read_variable_values(value_function.variable_registry)
        w1_value = var_dict["value-function/hidden-layer/dense/kernel"]
        b1_value = var_dict["value-function/hidden-layer/dense/bias"]
        w2_value = var_dict["value-function/value-function-output/dense/kernel"]
        b2_value = var_dict["value-function/value-function-output/dense/bias"]

        expected = dense_layer(dense_layer(input_, w1_value, b1_value), w2_value, b2_value)

        test.test(("call", input_), expected_outputs=expected, decimals=5)

        test.terminate()

    def test_simple_value_function_using_keras_style_assembly(self):
        # Space must contain batch dimension (otherwise, NNlayer will complain).
        space = FloatBox(shape=(3,), add_batch_rank=True)

        out1 = DenseLayer(units=3, scope="a")(space)
        out2 = DenseLayer(units=5, scope="b")(out1)

        # Create a simple neural net from json.
        value_function = ValueFunction(outputs=out2)

        # Do not seed, we calculate expectations manually.
        test = ComponentTest(component=value_function, input_spaces=dict(inputs=space))

        # Batch of size=n.
        input_ = space.sample(10)
        # Calculate output manually.
        var_dict = test.read_variable_values(value_function.variable_registry)
        w1_value = var_dict["value-function/a/dense/kernel"]
        b1_value = var_dict["value-function/a/dense/bias"]
        w2_value = var_dict["value-function/b/dense/kernel"]
        b2_value = var_dict["value-function/b/dense/bias"]
        w3_value = var_dict["value-function/value-function-output/dense/kernel"]
        b3_value = var_dict["value-function/value-function-output/dense/bias"]

        expected = dense_layer(
            dense_layer(dense_layer(input_, w1_value, b1_value), w2_value, b2_value), w3_value, b3_value
        )

        test.test(("call", input_), expected_outputs=expected, decimals=5)

        test.terminate()

    def test_add_layer_to_value_function(self):
        # For now, should not be allowed.
        # Space must contain batch dimension (otherwise, NNlayer will complain).
        space = FloatBox(shape=(3,), add_batch_rank=True)

        # Create a simple neural net from json.
        value_function = ValueFunction.from_spec(config_from_path("configs/test_simple_nn.json")) # type: ValueFunction
        # Add another layer to it.
        try:
            value_function.add_layer(DenseLayer(units=10, scope="last-layer"))
        except RLGraphError as e:
            print("Seeing expected RLGraphValueFunctionError: Cannot add a layer to a done ValueFunction component. "
                  "Test ok.")
            return
