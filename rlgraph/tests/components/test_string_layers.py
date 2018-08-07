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
from rlgraph.spaces import FloatBox, IntBox
from rlgraph.tests import ComponentTest
from rlgraph.utils.numpy import sigmoid


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
