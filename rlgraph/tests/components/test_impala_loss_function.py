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

import numpy as np
import unittest

from rlgraph.components.loss_functions.impala_loss_function import IMPALALossFunction
from rlgraph.spaces import *
from rlgraph.tests import ComponentTest


class TestIMPALALossFunction(unittest.TestCase):

    reward_space = FloatBox(add_batch_rank=True, add_time_rank=True, time_major=True)
    terminal_space = BoolBox(add_batch_rank=True, add_time_rank=True, time_major=True)
    values_space = FloatBox(shape=(1,), add_batch_rank=True, add_time_rank=True, time_major=True)
    loss_per_item_space = FloatBox(add_batch_rank=True)

    def test_impala_loss_function(self):
        action_space = IntBox(4, shape=(), add_batch_rank=True)
        impala_loss_function = IMPALALossFunction(discount=0.99)
        action_probs_space = FloatBox(shape=(4,), add_batch_rank=True, add_time_rank=True, time_major=True)

        test = ComponentTest(
            component=impala_loss_function,
            input_spaces=dict(
                logits_actions_pi=action_probs_space,
                action_probs_mu=action_probs_space,
                values=self.values_space,
                actions=action_space.with_extra_ranks(add_time_rank=True, time_major=True),
                rewards=self.reward_space,
                terminals=self.terminal_space,
                loss_per_item=self.loss_per_item_space
            ),
            action_space=action_space
        )

        # Batch of size=2.
        size = (15, 34)  # Some crazy time/batch combination.
        input_ = [
            action_probs_space.sample(size=size),  # logits actions pi
            action_probs_space.sample(size=size),  # action probs mu
            self.values_space.sample(size=size),  # values
            action_space.with_extra_ranks(add_time_rank=True, time_major=True).sample(size=size),  # actions
            self.reward_space.sample(size=size),  # rewards
            self.terminal_space.sample(size=size),  # terminals
        ]

        # Batch size=2 -> Expect 2 values in the `loss_per_item` out-Socket.
        expected_loss_per_item = np.array([
            3.8007843, 5.743328, 3.3092523, 1.7032332, 7.3653383, 4.096878, 3.7479136, 15.999393, 7.2431164, 6.3136196,
            -0.91453195, 13.271328, 6.89783, 3.9359467, 3.869079, 9.63856, 7.4453297, 7.601849, 7.215328, 5.6848574,
            4.21309, -0.36428893, -0.4812764, 3.1282873, 7.9603643, -2.5333924, 6.8146687, 8.46941, 0.10628, 1.7405777,
            1.8472002, 9.023342, 5.2372904, 3.293836
        ], dtype=np.float32)
        test.test(("loss_per_item", input_), expected_outputs=expected_loss_per_item, decimals=5)
        # Expect the mean over the batch.
        expected_loss = expected_loss_per_item.mean()
        test.test(("loss_average", expected_loss_per_item), expected_outputs=expected_loss, decimals=5)
        # Both.
        test.test(("loss", input_), expected_outputs=[expected_loss, expected_loss_per_item], decimals=5)

