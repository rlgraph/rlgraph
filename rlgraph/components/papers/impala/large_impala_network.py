# Copyright 2018 The RLgraph authors, All Rights Reserved.
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

from rlgraph.components.neural_networks import NeuralNetwork
from rlgraph.components.layers.nn.dense_layer import DenseLayer
from rlgraph.components.layers.nn.conv2d_layer import Conv2DLayer
from rlgraph.components.layers.nn.nn_layer import NNLayer
from rlgraph.components.layers.nn.residual_layer import ResidualLayer
from rlgraph.components.layers.nn.maxpool2d_layer import MaxPool2DLayer
from rlgraph.components.layers.nn.lstm_layer import LSTMLayer
from rlgraph.components.layers.nn.concat_layer import ConcatLayer
from rlgraph.components.layers.strings.string_to_hash_bucket import StringToHashBucket
from rlgraph.components.layers.strings.embedding_lookup import EmbeddingLookup
from rlgraph.components.neural_networks.stack import Stack
from rlgraph.components.common.repeater_stack import RepeaterStack
from rlgraph.components.common.splitter import Splitter


class LargeIMPALANetwork(NeuralNetwork):
    """
    The "large architecture" version of the network used in [1].

    [1] IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures - Espeholt, Soyer,
        Munos et al. - 2018 (https://arxiv.org/abs/1802.01561)
    """
    def __init__(self, scope="large-impala-network", **kwargs):
        super(LargeIMPALANetwork, self).__init__(scope=scope, **kwargs)

        # Create all needed sub-components.

        # Splitter for the Env signal (dict of 4 keys: for env image, env text, previous action and reward).
        self.splitter = Splitter("image", "text", "previous_action", "previous_reward")

        # The Image Processing Stack (left side of "Large Architecture" Figure 3 in [1]).
        # Conv2D column + ReLU + fc(256) + ReLU.
        self.image_processing_stack = self.build_image_processing_stack()

        # The text processing pipeline: Takes a batch of string tensors as input, creates a hash-bucket thereof,
        # and passes the output of the hash bucket through an embedding-lookup(20) layer. The output of the embedding
        # lookup is then passed through an LSTM(64).
        self.text_processing_stack = self.build_text_processing_stack()

        # The concatenation layer (concatenates outputs from image/text processing stacks, previous action/reward).
        self.concat_layer = ConcatLayer()

        self.main_lstm = LSTMLayer(units=256, scope="lstm-256")

        # Add all sub-components to this one.
        self.add_components(
            self.splitter, self.image_processing_stack, self.text_processing_stack, self.concat_layer, self.main_lstm
        )

        # TODO: Define the API method (apply) to stick it all together.

    @staticmethod
    def build_image_processing_stack():
        """
        Builds the 3 sequential Conv2D blocks that process the image information.
        Each of these 3 blocks consists of:
        - 1 Conv2D layer followed by a MaxPool2D
        - 2 residual blocks, each of which looks like:
            - ReLU + Conv2D + ReLU + Conv2D + element-wise add with original input
        """
        conv2d_main_units = list()
        for i, num_filters in enumerate([16, 32, 32]):
            # Conv2D plus MaxPool2D.
            conv2d_plus_maxpool = Stack(
                Conv2DLayer(filters=num_filters, kernel_size=3, strides=1),
                MaxPool2DLayer(pool_size=3, strides=2)
            )

            # Single unit for the residual layers (ReLU + Conv2D 3x3 stride=1).
            residual_unit = Stack(
                NNLayer(activation="relu"),  # single ReLU
                Conv2DLayer(filters=num_filters, kernel_size=3, strides=1)
            )
            # Residual Layer.
            residual_layer = ResidualLayer(residual_unit=residual_unit, repeats=2)
            # Repeat same residual layer 2x.
            residual_repeater = RepeaterStack(sub_component=residual_layer, repeats=2)

            conv2d_main_units.append(Stack(conv2d_plus_maxpool, residual_repeater))

        # Sequence together the conv2d units and an fc block (surrounded by ReLUs).
        return Stack(
            conv2d_main_units + [NNLayer(activation="relu"), DenseLayer(units=256), NNLayer(activation="relu")],
            scope="image-processing-stack"
        )

    @staticmethod
    def build_text_processing_stack():
        """
        Builds the text processing pipeline consisting of:
        - 1 StringToHashBucket Layer taking a batch of sentences and converting them to an indices-table of dimensions:
            cols=length of longest sentences in input
            rows=number of items in the batch
            The cols dimension could be interpreted as the time rank into a consecutive LSTM. The StringToHashBucket
            Component returns the sequence length of each batch item for exactly that purpose.
        - 1 Embedding Lookup Layer of embedding size 20 and number of rows == num_hash_buckets (see previous layer).
        - 1 LSTM processing the batched sequences of words coming from the embedding layer as batches of rows.
        """
        num_hash_buckets = 1000

        string_to_hash_bucket = StringToHashBucket(num_hash_buckets=num_hash_buckets)
        embedding = EmbeddingLookup(embed_dim=20, vocab_size=num_hash_buckets)

        lstm64 = LSTMLayer(units=64, scope="lstm-64")

        # TODO: Stack with LSTM (side) input of initial hidden state.
        # TODO: time-rank must be unfolded again from batch rank before passing into LSTM.
        return Stack(
            string_to_hash_bucket, embedding, lstm64,
            scope="text-processing-stack"
        )

    def apply(self, input_dict):
        # Split the input dict coming directly from the Env.
        # TODO: How do we get the previous action and reward in there?
        image, text, previous_action, previous_reward = self.call(self.splitter.split, input_dict)

        # Get the left-stack (image) and right-stack (text) output (see [1] for details).
        image_processing_output = self.call(self.image_processing_stack.apply, image)
        text_processing_output = self.call(self.text_processing_stack.apply, text)

        # Concat everything together.
        main_lstm_input = self.call(self.concat_layer.apply, image_processing_output, text_processing_output,
                                    previous_action, previous_reward)

        # Feed concat'd input into main LSTM(256).
        # TODO: initial hidden state (probably another input to this API-method)?
        action_probs, values = self.call(self.main_lstm.apply, main_lstm_input)

        return action_probs, values
