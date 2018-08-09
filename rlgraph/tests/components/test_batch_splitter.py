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
from rlgraph.components.common import Splitter, Merger
from rlgraph.spaces import *
from rlgraph.tests import ComponentTest


class TestBatchSplitter(unittest.TestCase):
    """
    Tests the Splitter- and Merger-Components.
    """
    def test_splitter_component(self):
        num_shards = 4
        space = Dict(
            states=dict(state1=float, state2=float),
            actions=dict(action1=float),
            reward=float,
            terminals=BoolBox(),
            add_batch_rank=True
        )

        sample = space.sample(size=21)

        test_inputs = [elem for elem in sample.values()]
        splitter = BatchSplitter(num_shards=num_shards)
        test = ComponentTest(component=splitter, input_spaces=dict(inputs=space))

        shards = test.test(("split_batch", test_inputs))
        print(shards)