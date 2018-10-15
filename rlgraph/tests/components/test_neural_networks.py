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
import unittest

from rlgraph.components.layers.nn.dense_layer import DenseLayer
from rlgraph.components.layers.nn.lstm_layer import LSTMLayer
from rlgraph.components.neural_networks import NeuralNetwork
from rlgraph.spaces import FloatBox, Tuple
from rlgraph.tests import ComponentTest
from rlgraph.utils.numpy import dense_layer, lstm_layer


class TestNeuralNetworks(unittest.TestCase):
    """
    Tests for assembling from json and running different NeuralNetworks.
    """
    def test_simple_nn(self):
        # Space must contain batch dimension (otherwise, NNlayer will complain).
        space = FloatBox(shape=(3,), add_batch_rank=True)

        # Create a simple neural net from json.
        neural_net = NeuralNetwork.from_spec("../configs/test_simple_nn.json")  # type: NeuralNetwork

        # Do not seed, we calculate expectations manually.
        test = ComponentTest(component=neural_net, input_spaces=dict(inputs=space))

        # Batch of size=3.
        input_ = np.array([[0.1, 0.2, 0.3], [1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
        # Calculate output manually.
        var_dict = neural_net.get_variables("hidden-layer/dense/kernel", "hidden-layer/dense/bias", global_scope=False)
        w1_value = test.read_variable_values(var_dict["hidden-layer/dense/kernel"])
        b1_value = test.read_variable_values(var_dict["hidden-layer/dense/bias"])

        expected = dense_layer(input_, w1_value, b1_value)

        test.test(("apply", input_), expected_outputs=dict(output=expected), decimals=5)

        test.terminate()

    def test_lstm_nn(self):
        # Space must contain batch dimension (otherwise, NNlayer will complain).
        units = 3
        batch_size = 2
        time_steps = 4
        input_nodes = 2
        input_space = FloatBox(shape=(input_nodes,), add_batch_rank=True, add_time_rank=True)
        internal_states_space = Tuple(FloatBox(shape=(units,)), FloatBox(shape=(units,)), add_batch_rank=True)

        def custom_apply(self, input_, internal_states=None):
            d0_out = self.get_sub_component_by_name("d0").apply(input_)
            lstm_out = self.get_sub_component_by_name("lstm").apply(d0_out, internal_states)
            d1_out = self.get_sub_component_by_name("d1").apply(lstm_out["output"])
            return dict(output=d1_out, last_internal_states=lstm_out["last_internal_states"])

        # Create a simple neural net with the above custom API-method.
        neural_net = NeuralNetwork(
            DenseLayer(units, scope="d0"),
            LSTMLayer(units, scope="lstm"),
            DenseLayer(units, scope="d1"),
            api_methods={("apply", custom_apply)}
        )

        # Do not seed, we calculate expectations manually.
        test = ComponentTest(component=neural_net, input_spaces=dict(
            input_=input_space, internal_states=internal_states_space
        ))

        # Batch of size=2, time-steps=3.
        input_ = input_space.sample((batch_size, time_steps))
        internal_states = internal_states_space.sample(batch_size)

        # Calculate output manually.
        w0_value = test.read_variable_values(neural_net.variables["neural-network/d0/dense/kernel"])
        b0_value = test.read_variable_values(neural_net.variables["neural-network/d0/dense/bias"])
        w1_value = test.read_variable_values(neural_net.variables["neural-network/d1/dense/kernel"])
        b1_value = test.read_variable_values(neural_net.variables["neural-network/d1/dense/bias"])
        lstm_w_value = test.read_variable_values(neural_net.variables["neural-network/lstm/lstm-cell/kernel"])
        lstm_b_value = test.read_variable_values(neural_net.variables["neural-network/lstm/lstm-cell/bias"])

        d0_out = dense_layer(input_, w0_value, b0_value)
        lstm_out, last_internal_states = lstm_layer(
            d0_out, lstm_w_value, lstm_b_value, initial_internal_states=internal_states, time_major=False
        )
        d1_out = dense_layer(lstm_out, w1_value, b1_value)

        expected = dict(output=d1_out, last_internal_states=last_internal_states)
        test.test(("apply", [input_, internal_states]), expected_outputs=expected, decimals=5)

        test.terminate()

    def test_dictionary_input_nn(self):
        pass
