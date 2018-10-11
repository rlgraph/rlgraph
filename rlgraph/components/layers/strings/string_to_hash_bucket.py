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
from rlgraph.components.layers.strings.string_layer import StringLayer
from rlgraph.spaces.space_utils import sanity_check_space
from rlgraph.utils.decorators import rlgraph_api
from rlgraph.utils.util import dtype as dtype_

if get_backend() == "tf":
    import tensorflow as tf


class StringToHashBucket(StringLayer):
    """
    A string to hash-bucket converter Component that takes a batch of string inputs (e.g.
    ["this is string A", "this is string B"] <- batch size==2) and creates a table of indices out of it that can be
    used instead of a static vocabulary list for embedding lookups. The created indices table contains
    n rows (n = number of items (strings) in the input batch) and m columns (m=max number of words in any of the
    input strings) of customizable int type.
    The int numbers in the created table can range from 0 to H (with H being the `num_hash_buckets` parameter).
    The entire hash bucket can now be fed through an embedding, producing - for each item in the batch - an m x e
    matrix, where m is the number of words in the batch item (sentence) (corresponds to an LSTM sequence length) and
    e is the embedding size. The embedding output can then be fed - e.g. - into an LSTM with m being the time rank
    (n still the batch rank).
    """
    def __init__(self, delimiter=" ", dtype="int64", num_hash_buckets=1000, hash_function="fast",
                 scope="string-to-hash-bucket", **kwargs):
        """
        Args:
            delimiter (str): The string delimiter used for splitting the input sentences into single "words".
                Default: " ".
            dtype (str): The type specifier for the created hash bucket. Default: int64.
            num_hash_buckets (int): The maximum value of any number in the created lookup table (lowest value
                is always 0).
            hash_function (str): The hashing function to use. One of "fast" or "strong". Default: "fast".
                For details, see: https://www.tensorflow.org/api_docs/python/tf/string_to_hash_bucket_(fast|strong)
                The "strong" method is better at avoiding placing different words into the same bucket, but runs
                about 4x slower than the "fast" one.

        Keyword Args:
            hash_keys (List[int,int]): Two uint64 keys used by the "strong" hashing function.
        """
        super(StringToHashBucket, self).__init__(scope=scope, **kwargs)

        self.delimiter = delimiter
        self.dtype = dtype
        assert self.dtype in ["int16", "int32", "int", "int64"],\
            "ERROR: dtype '{}' not supported by StringToHashBucket Component!".format(self.dtype)
        self.num_hash_buckets = num_hash_buckets
        self.hash_function = hash_function
        # Only used in the "strong" hash function.
        self.hash_keys = kwargs.pop("hash_keys", [12345, 67890])

    def check_input_spaces(self, input_spaces, action_space=None):
        super(StringToHashBucket, self).check_input_spaces(input_spaces, action_space)

        # Make sure there is only a batch rank (single text items).
        # tf.string_split does not support more complex shapes.
        sanity_check_space(input_spaces["text_inputs"], must_have_batch_rank=True, must_have_time_rank=False, rank=0)

    @rlgraph_api(flatten_ops=True, split_ops=True)
    def _graph_fn_apply(self, text_inputs):
        """
        Args:
            text_inputs (SingleDataOp): The Text input to generate a hash bucket for.

        Returns:
            tuple:
                - SingleDataOp: The hash lookup table (int64) that can be used as input to embedding-lookups.
                - SingleDataOp: The length (number of words) of the longest string in the `text_input` batch.
        """
        if get_backend() == "tf":
            # Split the input string.
            split_text_inputs = tf.string_split(source=text_inputs, delimiter=self.delimiter)
            # Build a tensor of n rows (number of items in text_inputs) words with
            dense = tf.sparse_tensor_to_dense(sp_input=split_text_inputs, default_value="")

            length = tf.reduce_sum(input_tensor=tf.to_int32(x=tf.not_equal(x=dense, y="")), axis=-1)
            if self.hash_function == "fast":
                hash_bucket = tf.string_to_hash_bucket_fast(input=dense, num_buckets=self.num_hash_buckets)
            else:
                hash_bucket = tf.string_to_hash_bucket_strong(input=dense,
                                                              num_buckets=self.num_hash_buckets,
                                                              key=self.hash_keys)

            # Int64 is tf's default for `string_to_hash_bucket` operation: Can leave as is.
            if self.dtype != "int64":
                hash_bucket = tf.cast(x=hash_bucket, dtype=dtype_(self.dtype))

            # Hash-bucket output is always batch-major.
            hash_bucket._batch_rank = 0
            hash_bucket._time_rank = 1

            return hash_bucket, length
