# Copyright 2018/2019 The RLgraph authors. All Rights Reserved.
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

        # Some crazy time/batch combination.
        size = (14, 34)
        size_state = (size[0]+1, size[1])  # States have one more slot (see env-stepper `step` output).
        input_ = [
            action_probs_space.sample(size=size_state),  # logits actions pi
            action_probs_space.sample(size=size),  # action probs mu
            self.values_space.sample(size=size_state),  # values
            action_space.with_extra_ranks(add_time_rank=True, time_major=True).sample(size=size),  # actions
            self.reward_space.sample(size=size),  # rewards
            self.terminal_space.sample(size=size),  # terminals
        ]

        # Batch size=2 -> Expect 2 values in the `loss_per_item` out-Socket.
        expected_loss_per_item = np.array([
            3.163107, 0.7833158, 4.313544, 4.540819, 2.8402734, 4.001171, 3.4556258, 4.4281297, 4.2689576, 7.52999,
            4.073623, 8.217271, 4.697727, 7.644105, 11.693808, 9.09968, 8.663768, 3.2075443, 6.819166, 4.017282,
            0.96279377, 2.6984437, 1.3387702, 5.492561, 3.0185113, -0.1206184, 5.096336, 5.011886, 2.627573, 3.4739096,
            6.6671352, 2.14344, 4.222275, 1.3773788
        ], dtype=np.float32)

        test.test(("loss_per_item", input_), expected_outputs=expected_loss_per_item, decimals=5)
        # Expect the mean over the batch.
        expected_loss = expected_loss_per_item.mean()
        test.test(("loss_average", expected_loss_per_item), expected_outputs=expected_loss, decimals=5)
        # Both.
        test.test(("loss", input_), expected_outputs=[expected_loss, expected_loss_per_item], decimals=5)

        test.terminate()

