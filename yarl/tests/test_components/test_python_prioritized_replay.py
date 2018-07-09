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
import numpy as np

from yarl.components.memories import PrioritizedReplay
from yarl.components.memories.mem_prioritized_replay import MemPrioritizedReplay
from yarl.spaces import Dict, IntBox, BoolBox, FloatBox
from yarl.tests import ComponentTest
from yarl.tests.test_util import non_terminal_records


class TestPythonPrioritizedReplay(unittest.TestCase):
    """
    Tests sampling and insertion behaviour of the mem_prioritized_replay module.
    """
    record_space = Dict(
        states=dict(state1=float, state2=float),
        actions=dict(action1=float),
        reward=float,
        terminals=BoolBox(),
        add_batch_rank=True
    )
    memory_variables = ["size", "index", "max-priority"]

    capacity = 10
    alpha = 1.0
    beta = 1.0

    max_priority = 1.0

    input_spaces = dict(
        # insert: records
        insert_records=record_space,
        # get_records: num_records
        get_records=int,
        # update_records: indices, update
        update_records=[IntBox(shape=(), add_batch_rank=True), FloatBox(shape=(), add_batch_rank=True)]
    )

    def test_insert(self):
        """
        Simply tests insert op without checking internal logic.
        """
        memory = MemPrioritizedReplay(
            capacity=self.capacity,
            next_states=True,
            alpha=self.alpha,
            beta=self.beta
        )
        memory.create_variables(self.input_spaces, None)

        observation = self.record_space.sample(size=1)
        memory.insert_records(observation)