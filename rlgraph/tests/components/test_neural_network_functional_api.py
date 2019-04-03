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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

from rlgraph.components.layers.nn.dense_layer import DenseLayer
from rlgraph.components.neural_networks import NeuralNetwork
from rlgraph.spaces import FloatBox
from rlgraph.tests.component_test import ComponentTest
from rlgraph.utils.numpy import dense_layer, relu


class TestNeuralNetworkFunctionalAPI(unittest.TestCase):
    """
    Tests for assembling from json and running different NeuralNetworks.
    """
    def test_functional_api_simple_nn(self):
        # Input Space of the network.
        input_space = FloatBox(shape=(3,), add_batch_rank=True)

        # Create a DenseLayer with a fixed `call` method input space for the arg `inputs`.
        output1 = DenseLayer(units=5, activation="linear", scope="a")(input_space)
        # Create a DenseLayer whose `inputs` arg is the resulting DataOpRec of output1's `call` output.
        output2 = DenseLayer(units=7, activation="relu", scope="b")(output1)

        # This will trace back automatically through the given output DataOpRec(s) and add all components
        # on the way to the input-space to this network.
        neural_net = NeuralNetwork(outputs=output2)

        # Do not seed, we calculate expectations manually.
        test = ComponentTest(component=neural_net, input_spaces=dict(nn_input=input_space))

        # Batch of size=3.
        input_ = input_space.sample(5)
        # Calculate output manually.
        var_dict = neural_net.get_variables()
        w1_value = test.read_variable_values(var_dict["neural-network/a/dense/kernel"])
        b1_value = test.read_variable_values(var_dict["neural-network/a/dense/bias"])
        w2_value = test.read_variable_values(var_dict["neural-network/b/dense/kernel"])
        b2_value = test.read_variable_values(var_dict["neural-network/b/dense/bias"])

        expected = relu(dense_layer(dense_layer(input_, w1_value, b1_value), w2_value, b2_value))

        test.test(("apply", input_), expected_outputs=dict(output=expected), decimals=5)

        test.terminate()


