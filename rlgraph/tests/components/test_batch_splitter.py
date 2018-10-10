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

from rlgraph.components import BatchSplitter
from rlgraph.spaces import *
from rlgraph.tests import ComponentTest


class TestBatchSplitter(unittest.TestCase):
    """
    Tests the BatchSplitter Component.
    """
    def test_batch_splitter_component(self):
        num_shards = 4
        space = Dict(
            states=dict(state1=float, state2=float),
            actions=dict(action1=float),
            rewards=float,
            terminals=BoolBox(),
            add_batch_rank=True
        )

        sample = space.sample(size=21)

        test_inputs = [sample["states"], sample["actions"], sample["rewards"], sample["terminals"]]
        splitter = BatchSplitter(num_shards=num_shards, shard_size=5)
        test = ComponentTest(component=splitter, input_spaces=dict(
            inputs=[space["states"], space["actions"], space["rewards"], space["terminals"]]
        ))

        # Expect 4 shards.
        expected = tuple(
            (
                dict(state1=sample["states"]["state1"][start:stop], state2=sample["states"]["state2"][start:stop]),
                dict(action1=sample["actions"]["action1"][start:stop]),
                sample["rewards"][start:stop],
                sample["terminals"][start:stop]
            ) for start, stop in [(0, 5), (5, 10), (10, 15), (15, 20)]
        )

        test.test(("split_batch", test_inputs), expected_outputs=expected)
