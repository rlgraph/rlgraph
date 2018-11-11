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

from rlgraph.components.layers.nn.dense_layer import DenseLayer
from rlgraph.components.layers.nn.conv2d_layer import Conv2DLayer
from rlgraph.components.layers.nn.nn_layer import NNLayer
from rlgraph.components.layers.nn.residual_layer import ResidualLayer
from rlgraph.components.layers.nn.maxpool2d_layer import MaxPool2DLayer
from rlgraph.components.layers.nn.lstm_layer import LSTMLayer
from rlgraph.components.layers.nn.concat_layer import ConcatLayer
from rlgraph.components.common.slice import Slice
from rlgraph.components.layers.preprocessing.multiply_divide import Divide
from rlgraph.components.layers.preprocessing.reshape import ReShape
from rlgraph.components.layers.preprocessing.image_crop import ImageCrop
from rlgraph.components.layers.strings.string_to_hash_bucket import StringToHashBucket
from rlgraph.components.layers.strings.embedding_lookup import EmbeddingLookup
from rlgraph.components.neural_networks.stack import Stack
from rlgraph.components.neural_networks.neural_network import NeuralNetwork
from rlgraph.components.common.repeater_stack import RepeaterStack
from rlgraph.components.common.container_splitter import ContainerSplitter
from rlgraph.utils.ops import DataOpTuple


class IMPALANetwork(NeuralNetwork):
    """
    The base class for both "large and small architecture" versions of the networks used in [1].

    [1] IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures - Espeholt, Soyer,
        Munos et al. - 2018 (https://arxiv.org/abs/1802.01561)
    """
    def __init__(self, worker_sample_size=100, scope="impala-network", **kwargs):
        """
        Args:
            worker_sample_size (int): How many time-steps an IMPALA actor will have performed in one rollout.
        """
        super(IMPALANetwork, self).__init__(scope=scope, **kwargs)

        self.worker_sample_size = worker_sample_size

        # Create all needed sub-components.

        # ContainerSplitter for the Env signal (dict of 4 keys: for env image, env text, previous action and reward).
        self.splitter = ContainerSplitter("RGB_INTERLEAVED", "INSTR", "previous_action", "previous_reward",
                                          scope="input-splitter")

        # Fold the time rank into the batch rank.
        self.time_rank_fold_before_lstm = ReShape(fold_time_rank=True, scope="time-rank-fold-before-lstm")
        self.time_rank_unfold_before_lstm = ReShape(unfold_time_rank=True, time_major=True,
                                                    scope="time-rank-unfold-before-lstm")

        # The Image Processing Stack (left side of "Large Architecture" Figure 3 in [1]).
        # Conv2D column + ReLU + fc(256) + ReLU.
        self.image_processing_stack = self.build_image_processing_stack()

        # The text processing pipeline: Takes a batch of string tensors as input, creates a hash-bucket thereof,
        # and passes the output of the hash bucket through an embedding-lookup(20) layer. The output of the embedding
        # lookup is then passed through an LSTM(64).
        self.text_processing_stack = self.build_text_processing_stack()

        #self.debug_slicer = Slice(scope="internal-states-slicer", squeeze=True)

        # The concatenation layer (concatenates outputs from image/text processing stacks, previous action/reward).
        self.concat_layer = ConcatLayer()

        # The main LSTM (going into the ActionAdapter (next in the Policy Component that uses this NN Component)).
        # Use time-major as it's faster (say tf docs).
        self.main_lstm = LSTMLayer(units=256, scope="lstm-256", time_major=True, static_loop=self.worker_sample_size)

        # Add all sub-components to this one.
        self.add_components(
            self.splitter, self.image_processing_stack, self.text_processing_stack,
            self.concat_layer,
            self.main_lstm,
            self.time_rank_fold_before_lstm, self.time_rank_unfold_before_lstm,
            #self.debug_slicer
        )

    @staticmethod
    def build_image_processing_stack():
        """
        Builds the image processing pipeline for IMPALA and returns it.
        """
        raise NotImplementedError

    @staticmethod
    def build_text_processing_stack():
        """
        Helper function to build the text processing pipeline for both the large and small architectures, consisting of:
        - ReShape preprocessor to fold the incoming time rank into the batch rank.
        - StringToHashBucket Layer taking a batch of sentences and converting them to an indices-table of dimensions:
          cols=length of longest sentences in input
          rows=number of items in the batch
          The cols dimension could be interpreted as the time rank into a consecutive LSTM. The StringToHashBucket
          Component returns the sequence length of each batch item for exactly that purpose.
        - Embedding Lookup Layer of embedding size 20 and number of rows == num_hash_buckets (see previous layer).
        - LSTM processing the batched sequences of words coming from the embedding layer as batches of rows.
        """
        num_hash_buckets = 1000

        # Create a hash bucket from the sentences and use that bucket to do an embedding lookup (instead of
        # a vocabulary).
        string_to_hash_bucket = StringToHashBucket(num_hash_buckets=num_hash_buckets)
        embedding = EmbeddingLookup(embed_dim=20, vocab_size=num_hash_buckets, pad_empty=True)
        # The time rank for the LSTM is now the sequence of words in a sentence, NOT the original env time rank.
        # We will only use the last output of the LSTM-64 for further processing as that is the output after having
        # seen all words in the sentence.
        # The original env stepping time rank is currently folded into the batch rank and must be unfolded again before
        # passing it into the main LSTM.
        lstm64 = LSTMLayer(units=64, scope="lstm-64", time_major=False)

        tuple_splitter = ContainerSplitter(tuple_length=2, scope="tuple-splitter")

        def custom_apply(self, inputs):
            hash_bucket, lengths = self.sub_components["string-to-hash-bucket"].apply(inputs)

            embedding_output = self.sub_components["embedding-lookup"].apply(hash_bucket)

            # Return only the last output (sentence of words, where we are not interested in intermediate results
            # where the LSTM has not seen the entire sentence yet).
            # Last output is the final internal h-state (slot 1 in the returned LSTM tuple; slot 0 is final c-state).
            _, lstm_final_internals = self.sub_components["lstm-64"].apply(embedding_output, sequence_length=lengths)

            # Need to split once more because the LSTM state is always a tuple of final c- and h-states.
            _, lstm_final_h_state = self.sub_components["tuple-splitter"].split(lstm_final_internals)

            return lstm_final_h_state

        text_processing_stack = Stack(
            string_to_hash_bucket, embedding, lstm64, tuple_splitter,
            api_methods={("apply", custom_apply)}, scope="text-stack"
        )

        return text_processing_stack

    def apply(self, input_dict, internal_states=None):
        # Split the input dict coming directly from the Env.
        _, _, _, orig_previous_reward = self.splitter.split(input_dict)

        folded_input = self.time_rank_fold_before_lstm.apply(input_dict)
        image, text, previous_action, previous_reward = self.splitter.split(folded_input)

        # Get the left-stack (image) and right-stack (text) output (see [1] for details).
        text_processing_output = self.text_processing_stack.apply(text)
        image_processing_output = self.image_processing_stack.apply(image)

        # Concat everything together.
        concatenated_data = self.concat_layer.apply(
            image_processing_output, text_processing_output, previous_action, previous_reward
        )

        unfolded_concatenated_data = self.time_rank_unfold_before_lstm.apply(concatenated_data, orig_previous_reward)

        # Feed concat'd input into main LSTM(256).
        main_lstm_output, main_lstm_final_c_and_h = self.main_lstm.apply(unfolded_concatenated_data, internal_states)

        return main_lstm_output, main_lstm_final_c_and_h


class LargeIMPALANetwork(IMPALANetwork):

    @staticmethod
    def build_image_processing_stack():
        """
        Constructs a ReShape preprocessor to fold the time rank into the batch rank.

        Then builds the 3 sequential Conv2D blocks that process the image information.
        Each of these 3 blocks consists of:
        - 1 Conv2D layer followed by a MaxPool2D
        - 2 residual blocks, each of which looks like:
            - ReLU + Conv2D + ReLU + Conv2D + element-wise add with original input

        Then adds: ReLU + fc(256) + ReLU.
        """
        # Collect components for image stack before unfolding time-rank going into main LSTM.
        sub_components = list()

        # Divide by 255
        sub_components.append(Divide(divisor=255, scope="divide-255"))

        for i, num_filters in enumerate([16, 32, 32]):
            # Conv2D plus MaxPool2D.
            conv2d_plus_maxpool = Stack(
                Conv2DLayer(filters=num_filters, kernel_size=3, strides=1, padding="same"),
                MaxPool2DLayer(pool_size=3, strides=2, padding="same"),
                scope="conv-max"
            )

            # Single unit for the residual layers (ReLU + Conv2D 3x3 stride=1).
            residual_unit = Stack(
                NNLayer(activation="relu"),  # single ReLU
                Conv2DLayer(filters=num_filters, kernel_size=3, strides=1, padding="same"),
                scope="relu-conv"
            )
            # Residual Layer.
            residual_layer = ResidualLayer(residual_unit=residual_unit, repeats=2)
            # Repeat same residual layer 2x.
            residual_repeater = RepeaterStack(sub_component=residual_layer, repeats=2)

            sub_components.append(Stack(conv2d_plus_maxpool, residual_repeater, scope="conv-unit-{}".format(i)))

        # A Flatten preprocessor and then an fc block (surrounded by ReLUs) and a time-rank-unfolding.
        sub_components.extend([
            ReShape(flatten=True, scope="flatten"),  # Flattener (to flatten Conv2D output for the fc layer).
            NNLayer(activation="relu", scope="relu-1"),  # ReLU 1
            DenseLayer(units=256),  # Dense layer.
            NNLayer(activation="relu", scope="relu-2"),  # ReLU 2
        ])

        image_stack= Stack(sub_components, scope="image-stack")

        return image_stack


class SmallIMPALANetwork(IMPALANetwork):
    """
    The "small/shallow architecture" version of the network used in [1].

    [1] IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures - Espeholt, Soyer,
        Munos et al. - 2018 (https://arxiv.org/abs/1802.01561)
    """

    @staticmethod
    def build_image_processing_stack():
        """
        Constructs a ReShape preprocessor to fold the time rank into the batch rank.

        Then builds the 2 Conv2D Layers followed by ReLUs.

        Then adds: fc(256) + ReLU.
        """
        # Collect components for image stack before unfolding time-rank going into main LSTM.
        sub_components = list()

        # Divide by 255
        sub_components.append(Divide(divisor=255, scope="divide-255"))

        for i, (num_filters, kernel_size, stride) in enumerate(zip([16, 32], [8, 4], [4, 2])):
            # Conv2D plus ReLU activation function.
            conv2d = Conv2DLayer(
                filters=num_filters, kernel_size=kernel_size, strides=stride, padding="same",
                activation="relu", scope="conv2d-{}".format(i)
            )
            sub_components.append(conv2d)

        # A Flatten preprocessor and then an fc block (surrounded by ReLUs) and a time-rank-unfolding.
        sub_components.extend([
            ReShape(flatten=True, scope="flatten"),  # Flattener (to flatten Conv2D output for the fc layer).
            DenseLayer(units=256),  # Dense layer.
            NNLayer(activation="relu", scope="relu-before-lstm"),
        ])

        #stack_before_unfold = <- formerly known as
        image_stack = Stack(sub_components, scope="image-stack")

        return image_stack
