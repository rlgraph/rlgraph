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

if get_backend() == "tf":
    import tensorflow as tf


class StringToHashBucket(Layer):
    """
    A string to hash-bucket converter Component that takes a batch of string inputs (e.g.
    ["this is string A", "this is string B"]) and creates a lookup table out of it that can be used instead of a
    static vocabulary list for embedding lookups. The created lookup table contains n rows (n = number of items
    (strings) in the input batch) and m columns (m=max number of words in any of the input strings).
    The int numbers in the created lookup table can range from 0 to H (with H being the `num_hash_buckets` parameter).
    """
    # TODO: add options: delimiter (split), etc...
    def __init__(self, num_hash_buckets=1000, scope="str-to-hash-bucket", **kwargs):
        """
        Args:
            num_hash_buckets (int): The maximum value in the created lookup table.
        """
        super(StringToHashBucket, self).__init__(scope=scope, **kwargs)

        self.num_hash_buckets = num_hash_buckets

    def check_input_spaces(self, input_spaces, action_space=None):
        super(StringToHashBucket, self).check_input_spaces(input_spaces, action_space)

        # Make sure there is only a batch rank (single text items).
        # tf.string_split does not support more complex shapes.
        sanity_check_space(input_spaces["text_input"], must_have_batch_rank=True, rank=0)

    def _graph_fn_apply(self, text_input):
        """
        Args:
            text_input (SingleDataOp): The Text input to generate a hash bucket for.

        Returns:
            tuple:
                - SingleDataOp: The hash lookup table that can be used as input to embedding-lookups.
                - SingleDataOp: The length (number of words) of the longest string in the `text_input` batch.
        """
        if get_backend() == "tf":
            # Split the input string.
            split_text_input = tf.string_split(text_input)
            dense = tf.sparse_tensor_to_dense(split_text_input, default_value="")
            length = tf.reduce_sum(tf.to_int32(tf.not_equal(dense, "")), axis=-1)

            # To int64 hash buckets. Small risk of having collisions. Alternatively, a
            # vocabulary can be used.
            return tf.string_to_hash_bucket_fast(dense, self.num_hash_buckets), length
