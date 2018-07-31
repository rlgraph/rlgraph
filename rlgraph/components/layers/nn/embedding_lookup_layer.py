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

from rlgraph import get_backend
from rlgraph.utils.util import dtype
from rlgraph.components.layers.nn.nn_layer import NNLayer

if get_backend() == "tf":
    import tensorflow as tf


class EmbeddingLookupLayer(NNLayer):
    """
    An embedding lookup layer.
    A matrix with num-columns = number of encoding value per vocab and num-rows = number of vocabs to encode.
    Calling `apply` will lookup and return rows from this matrix specified via the input to `apply` as a simple
    tensor of row indices.
    """
    def __init__(self, embed_dim, vocab_size, partition_strategy="mod", trainable=True, **kwargs):
        """
        Args:
            embed_dim (int): The number of values (number of columns) to use for the encoding of each vocab. One vocab
                equals one row in the embedding matrix.
            vocab_size (int): The number of vocabs (number of rows) in the embedding matrix.
            partition_strategy (str): One of "mod" or "div". Default: "mod".
            trainable (bool): Whether the Variable(s) representing the embedding matrix should be trainable or not.
                Default: True.
        """
        super(EmbeddingLookupLayer, self).__init__(scope=kwargs.pop("scope", "embedding-lookup"), **kwargs)

        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.partition_strategy = partition_strategy
        self.trainable = trainable

        # Our embedding matrix variable.
        self.embedding_matrix = None

    def create_variables(self, input_spaces, action_space):
        # TODO: For IMPALA, neither partitioner, nor initializer are needed. Do these later.
        self.embedding_matrix = self.get_variable(shape=(self.vocab_size, self.embed_dim), dtype=dtype("float"),
                                                  #initializer=self.initializers,
                                                  #partitioner=self.partitioners,
                                                  #regularizer=self.regularizers,
                                                  trainable=self.trainable)

    def _graph_fn_apply(self, ids):
        if get_backend() == "tf":
            return tf.nn.embedding_lookup(
                self.embedding_matrix, ids, partition_strategy=self.partition_strategy, max_norm=None
            )
