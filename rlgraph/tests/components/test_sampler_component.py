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

from rlgraph.components import Sampler
from rlgraph.spaces import BoolBox, Dict
from rlgraph.tests import ComponentTest


class TestSamplerComponent(unittest.TestCase):
    """
    Tests the sampler component.
    """
    def test_sampler_component(self):
        input_space = Dict(
            states=dict(state1=float, state2=float),
            actions=dict(action1=float),
            reward=float,
            terminals=BoolBox(),
            add_batch_rank=True
        )

        sampler = Sampler()
        test = ComponentTest(component=sampler, input_spaces=dict(sample_size=int, inputs=input_space))

        samples = input_space.sample(size=100)
        sample = test.test(("sample", [10, samples]), expected_outputs=None)

        self.assertEqual(len(sample["actions"]["action1"]), 10)
        self.assertEqual(len(sample["states"]["state1"]), 10)
        self.assertEqual(len(sample["terminals"]), 10)

        print(sample)
