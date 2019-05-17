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

import unittest
from math import log

import numpy as np
from rlgraph.components.loss_functions import PPOLossFunction
from rlgraph.spaces import *
from rlgraph.tests import ComponentTest


class TestPPOLossFunctions(unittest.TestCase):

    input_spaces = dict(
        loss_per_item=FloatBox(add_batch_rank=True),
        log_probs=FloatBox(shape=(1,), add_batch_rank=True),
        prev_log_probs=FloatBox(shape=(1,), add_batch_rank=True),
        state_values=FloatBox(shape=(1,), add_batch_rank=True),
        prev_state_values=FloatBox(shape=(1,), add_batch_rank=True),
        advantages=FloatBox(add_batch_rank=True),
        entropy=FloatBox(add_batch_rank=True)
    )

    def test_ppo_loss_function_on_int_action_space(self):
        action_space = IntBox(2, add_batch_rank=True)
        clip_ratio = 0.2

        ppo_loss_function = PPOLossFunction(clip_ratio=clip_ratio, value_function_clipping=False)

        test = ComponentTest(component=ppo_loss_function, input_spaces=self.input_spaces, action_space=action_space)

        # Batch of size=n.
        log_probs = np.array([[log(0.4)], [log(0.9)], [log(0.1)]])
        prev_log_probs = np.array([[log(0.3)], [log(0.95)], [log(0.2)]])
        state_values = np.array([[-2.0], [-1.0], [1.0]])
        prev_state_values = np.array([[-3.4], [-1.3], [0.3]])
        advantages = np.array([1.0, 3.0, 2.0])
        entropy = np.array([0.7, 0.3, 3.2])

        """
        Calculation of PG loss term:
        # IS ratios
        rhos = probs / prev_probs = exp(log(probs/prev_probs)) = exp(log_probs - prev_log_probs)
        # clipping around 1.0
        clipped = clip(rhos, 1.0-clip_ratio, 1.0+clip_ratio)
        # entropy loss term
        Le = - weight * entropy
        
        L = min(clipped * A, rhos * A) + Le 
        """
        rhos = np.exp(log_probs - prev_log_probs)
        clipped_rhos = np.clip(rhos, 1.0 - clip_ratio, 1.0 + clip_ratio)
        expanded_advantages = np.expand_dims(advantages, axis=-1)
        clipped_advantages = -np.minimum(rhos * expanded_advantages, clipped_rhos * expanded_advantages)
        entropy_term = -0.00025 * np.expand_dims(entropy, axis=-1)  # 0.00025 == default entropy weight

        expected_pg_loss_per_item = np.squeeze(clipped_advantages + entropy_term)

        test.test(
            ("pg_loss_per_item", [log_probs, prev_log_probs, advantages, entropy]),
            expected_outputs=expected_pg_loss_per_item, decimals=5
        )

        v_targets = advantages + np.squeeze(prev_state_values)  # Q-value targets
        expected_value_loss_per_item = np.square(np.squeeze(state_values) - v_targets)

        test.test(
            ("value_function_loss_per_item", [state_values, prev_state_values, advantages]),
            expected_outputs=expected_value_loss_per_item, decimals=5
        )

        # All together.
        test.test(
            ("loss_per_item", [log_probs, prev_log_probs, state_values, prev_state_values, advantages, entropy]),
            expected_outputs=[expected_pg_loss_per_item, expected_value_loss_per_item], decimals=5
        )

        # Expect the mean over the batch.
        test.test(("loss_average", expected_pg_loss_per_item), expected_outputs=expected_pg_loss_per_item.mean())

        # Both.
        test.test(
            ("loss", [log_probs, prev_log_probs, state_values, prev_state_values, advantages, entropy]),
            expected_outputs=[
                expected_pg_loss_per_item.mean(), expected_pg_loss_per_item,
                expected_value_loss_per_item.mean(), expected_value_loss_per_item
            ], decimals=5
        )

