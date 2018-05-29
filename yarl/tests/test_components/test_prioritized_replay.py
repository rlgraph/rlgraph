# Copyright 2018 The YARL-Project, All Rights Reserved.
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

from yarl.components.memories import PrioritizedReplay
from yarl.spaces import Dict, IntBox, FloatBox
from yarl.tests import ComponentTest
from yarl.tests.test_util import non_terminal_records, terminal_records


class TestPrioritizedReplay(unittest.TestCase):
    """
    Tests sampling and insertion behaviour of the prioritized_replay module.
    """
    record_space = Dict(
        states=dict(state1=float, state2=float),
        actions=dict(action1=float),
        reward=float,
        terminal=IntBox(low=0, high=1),
        add_batch_rank=True
    )
    memory_variables = ["size", "index"]
    capacity = 10

    def test_insert(self):
        """
        Simply tests insert op without checking internal logic.
        """
        memory = PrioritizedReplay(
            capacity=self.capacity,
            next_states=True
        )
        test = ComponentTest(component=memory, input_spaces=dict(
            records=self.record_space,
            num_records=int,
            indices=IntBox(shape=(), add_batch_rank=True),
            update=FloatBox(shape=(), add_batch_rank=True)
        ))

        observation = self.record_space.sample(size=1)
        test.test(out_socket_name="insert", inputs=observation, expected_outputs=None)