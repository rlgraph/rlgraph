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
from rlgraph.components.layers.layer import Layer
from rlgraph.spaces.space_utils import sanity_check_space
from rlgraph.utils.decorators import rlgraph_api
from rlgraph.utils.initializer import Initializer

if get_backend() == "tf":
    import tensorflow as tf


class EmbeddingLookup(Layer):
    """
    An embedding lookup layer.
    A matrix with num-columns = number of encoding value per vocab and num-rows = number of vocabs to encode.
    Calling `apply` will lookup and return rows from this matrix specified via the input to `apply` as a simple
    tensor of row indices.
    """
    def __init__(self, embed_dim, vocab_size, initializer_spec="truncated_normal", partition_strategy="mod",
                 trainable=True, pad_empty=False, **kwargs):
        """
        Args:
            embed_dim (int): The number of values (number of columns) to use for the encoding of each vocab. One vocab
                equals one row in the embedding matrix.
            vocab_size (int): The number of vocabs (number of rows) in the embedding matrix.
            initializer_spec (any): A specifier for the embedding matrix initializer.
                If None, use the default initializer, which is truncated normal with stddev=1/sqrt(vocab_size).
            partition_strategy (str): One of "mod" or "div". Default: "mod".
            trainable (bool): Whether the Variable(s) representing the embedding matrix should be trainable or not.
                Default: True.
            pad_empty (bool): Whether to pad the output if no lookups take place and the embedding would otherwise
                return a shape=(embed_dim, 0) output. If True, would then return shape=(embed_dim, 1).

        """
        super(EmbeddingLookup, self).__init__(scope=kwargs.pop("scope", "embedding-lookup"), **kwargs)

        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.initializer_spec = initializer_spec
        self.initializer = None
        self.partition_strategy = partition_strategy
        self.trainable = trainable
        self.pad_empty = pad_empty

        # Our embedding matrix variable.
        self.embedding_matrix = None

        self.ids_space = None

    def check_input_spaces(self, input_spaces, action_space=None):
        ids_space = input_spaces["ids"]
        # For now, require both batch- and time-ranks.
        sanity_check_space(ids_space, must_have_batch_rank=True, must_have_time_rank=True)

    def create_variables(self, input_spaces, action_space=None):
        # Create weights matrix and (maybe) biases vector.
        shape = (self.vocab_size, self.embed_dim)
        self.initializer = Initializer.from_spec(shape=shape, specification=self.initializer_spec)
        # TODO: For IMPALA partitioner is not needed. Do this later.
        self.embedding_matrix = self.get_variable(
            name="embedding-matrix", shape=shape, dtype=dtype("float"), initializer=self.initializer.initializer,
            #partitioner=self.partitioners, regularizer=self.regularizers,
            trainable=self.trainable
        )

        self.ids_space = input_spaces["ids"]

    @rlgraph_api(flatten_ops=True, split_ops=True)
    def _graph_fn_apply(self, ids):
        if get_backend() == "tf":
            embedding_lookup_output = tf.nn.embedding_lookup(
                self.embedding_matrix, ids, partition_strategy=self.partition_strategy, max_norm=None
            )
            # Do we need to CONSTANT-pad (with 0s) if empty?
            if self.pad_empty:
                padding = tf.to_int32(tf.equal(tf.shape(embedding_lookup_output)[1], 0))
                embedding_lookup_output = tf.pad(embedding_lookup_output, [[0, 0], [0, padding], [0, 0]])

            embedding_lookup_output._batch_rank = 0 if self.ids_space.time_major is False else 1
            embedding_lookup_output._time_rank = 0 if self.ids_space.time_major is True else 1
            return embedding_lookup_output
