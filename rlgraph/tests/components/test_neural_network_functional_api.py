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

from rlgraph.components.layers.nn import DenseLayer, LSTMLayer, ConcatLayer, Conv2DLayer
from rlgraph.components.layers.preprocessing.reshape import ReShape
from rlgraph.components.layers.strings import StringToHashBucket, EmbeddingLookup
from rlgraph.components.neural_networks import NeuralNetwork
from rlgraph.spaces import Dict, FloatBox, TextBox
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

        test = ComponentTest(component=neural_net, input_spaces=dict(inputs=input_space))

        # Batch of size=n.
        input_ = input_space.sample(5)
        # Calculate output manually.
        var_dict = neural_net.get_variables("a/dense/kernel", "a/dense/bias", "b/dense/kernel", "b/dense/bias", global_scope=False)
        w1_value = test.read_variable_values(var_dict["a/dense/kernel"])
        b1_value = test.read_variable_values(var_dict["a/dense/bias"])
        w2_value = test.read_variable_values(var_dict["b/dense/kernel"])
        b2_value = test.read_variable_values(var_dict["b/dense/bias"])

        expected = relu(dense_layer(dense_layer(input_, w1_value, b1_value), w2_value, b2_value))

        test.test(("apply", input_), expected_outputs=expected, decimals=5)

        test.terminate()

    def test_functional_api_multi_stream_nn(self):
        # Input Space of the network.
        input_space = Dict({
            "img": FloatBox(shape=(6, 6, 3)),  # some RGB img
            "txt": TextBox()  # some text
        }, add_batch_rank=True, add_time_rank=True)

        # Complex NN assembly via our Keras-style functional API.
        # Fold text input into single batch rank.
        folded_text = ReShape(fold_time_rank=True)(input_space["txt"])
        # String layer will create batched AND time-ranked (individual words) hash outputs (int64).
        string_bucket_out, lengths = StringToHashBucket(num_hash_buckets=5)(folded_text)
        # Batched and time-ranked embedding output (floats) with embed dim=n.
        embedding_out = EmbeddingLookup(embed_dim=10, vocab_size=5)(string_bucket_out)
        # Pass embeddings through a text LSTM and use last output (reduce time-rank).
        string_lstm_out = LSTMLayer(units=2, return_sequences=False)(embedding_out, sequence_length=lengths)
        # Unfold to get original time-rank back.
        # TODO: Let NN handle orig. input passing for unfolding into this layer.
        string_lstm_out_unfolded = ReShape(unfold_time_rank=True)(string_lstm_out)

        # Parallel image stream via 1 CNN layer plus dense.
        cnn_out = Conv2DLayer(filters=1, kernel_size=2, strides=2)(input_space["img"])
        dense_out = DenseLayer(units=2)(cnn_out)

        # Concat everything.
        concat_out = ConcatLayer()(string_lstm_out_unfolded, dense_out)

        # LSTM output has batch+time.
        main_lstm_out = LSTMLayer(units=2)(concat_out)

        dense_after_lstm_out = DenseLayer(units=3)(main_lstm_out)

        # A NN with 2 outputs.
        neural_net = NeuralNetwork(outputs=[dense_after_lstm_out, main_lstm_out])

        # Do not seed, we calculate expectations manually.
        test = ComponentTest(component=neural_net, input_spaces=dict(nn_input=input_space))

        # Batch of size=n.
        input_ = input_space.sample(5)
        # Calculate output manually.
        var_dict = neural_net.get_variables()
        w1_value = test.read_params("neural-network/a/dense/kernel", var_dict)
        b1_value = test.read_params("neural-network/a/dense/bias", var_dict)
        w2_value = test.read_params("neural-network/b/dense/kernel", var_dict)
        b2_value = test.read_params("neural-network/b/dense/bias", var_dict)

        expected = relu(dense_layer(dense_layer(input_, w1_value, b1_value), w2_value, b2_value))

        test.test(("apply", input_), expected_outputs=dict(output=expected), decimals=5)

        test.terminate()
