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

import unittest
import numpy as np

from rlgraph.components.layers.strings import *
from rlgraph.spaces import IntBox, TextBox
from rlgraph.tests import ComponentTest


class TestStringLayers(unittest.TestCase):
    """
    Tests for the different StringLayer Components. Each layer is tested separately.
    """
    def test_embedding_lookup_layer(self):
        # Input space for lookup indices (double indices for picking 2 rows per batch item).
        input_space = IntBox(shape=(2,), add_batch_rank=True)

        embedding = EmbeddingLookup(embed_dim=5, vocab_size=4, initializer_spec=np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
            [11.0, 12.0, 13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0, 19.0, 20.0]
        ]))
        test = ComponentTest(component=embedding, input_spaces=dict(ids=input_space))

        # Pull a batch of 3 (2 vocabs each) from the embedding matrix.
        inputs = np.array(
            [[0, 1], [3, 2], [2, 1]]
        )

        expected = np.array([
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0, 9.0, 10.0]
            ], [
                [16.0, 17.0, 18.0, 19.0, 20.0],
                [11.0, 12.0, 13.0, 14.0, 15.0]
            ], [
                [11.0, 12.0, 13.0, 14.0, 15.0],
                [6.0, 7.0, 8.0, 9.0, 10.0],
            ]
        ])
        test.test(("apply", inputs), expected_outputs=expected, decimals=5)

    def test_string_to_hash_bucket_layer(self):
        # Input space: Batch of strings.
        input_space = TextBox(add_batch_rank=True)

        # Use a fast-hash function with 10 possible buckets to put a word into.
        string_to_hash_bucket = StringToHashBucket(num_hash_buckets=10, hash_function="fast")
        test = ComponentTest(component=string_to_hash_bucket, input_spaces=dict(text_inputs=input_space))

        # Send a batch of 3 strings through the hash-bucket generator.
        inputs = np.array([
            "text A",
            "test B",
            "text C  D and E"
        ])

        # NOTE that some different words occupy the same hash bucket (e.g. 'C' and 'and' (7) OR 'text' and [empty] (3)).
        # This can be avoided by 1) picking a larger `num_hash_buckets` or 2) using the "strong" hash function.
        expected_hash_bucket = np.array([
            [3, 4, 3, 3, 3],  # text A .  .  .
            [6, 8, 3, 3, 3],  # test B .  .  .
            [3, 7, 5, 7, 2],  # text C D and E
        ])
        expected_lengths = np.array([2, 2, 5])
        test.test(("apply", inputs), expected_outputs=(expected_hash_bucket, expected_lengths))

    def test_string_to_hash_bucket_layer_with_different_ctor_params(self):
        # Input space: Batch of strings.
        input_space = TextBox(add_batch_rank=True)

        # Construct a strong hash bucket with different delimiter, larger number of buckets, string algo and
        # int16 dtype.
        string_to_hash_bucket = StringToHashBucket(delimiter="-", num_hash_buckets=20, hash_function="strong",
                                                   dtype="int16")
        test = ComponentTest(component=string_to_hash_bucket, input_spaces=dict(text_inputs=input_space))

        # Send a batch of 5 strings through the hash-bucket generator.
        inputs = np.array([
            "text-A",
            "test-B",
            "text-C--D-and-E",
            "bla bla-D"
        ])

        # NOTE that some different words occupy the same hash bucket (e.g. 'C' and 'and' OR 'text' and [empty]).
        # This can be avoided by 1) picking a larger `num_hash_buckets` or 2) using the "strong" hash function.
        expected_hash_bucket = np.array([
            [2, 6, 18, 18, 18],    # text    A .  .  .
            [12, 7, 18, 18, 18],   # test    B .  .  .
            [2, 6, 13, 19, 15],    # text    C D and E
            [13, 13, 18, 18, 18],  # bla bla D .  .  .  <- Note that "bla bla" and "D" still have the same bucket (13)
        ])
        expected_lengths = np.array([2, 2, 5, 2])
        test.test(("apply", inputs), expected_outputs=(expected_hash_bucket, expected_lengths))
