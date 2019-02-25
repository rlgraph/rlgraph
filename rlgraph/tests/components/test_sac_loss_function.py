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
import logging
import numpy as np

from rlgraph.agents.sac_agent import SACLossFunction
from rlgraph.spaces import FloatBox, BoolBox, Tuple, IntBox
from rlgraph.tests import ComponentTest
from rlgraph.utils import root_logger


class TestSACLossFunction(unittest.TestCase):
    """
    Tests the SAC Agent's functionality.
    """
    root_logger.setLevel(level=logging.DEBUG)

    @staticmethod
    def _prepare_loss_function_test(loss_function):
        test = ComponentTest(
            component=loss_function,
            input_spaces=dict(
                alpha=float,
                log_probs_next_sampled=FloatBox(shape=(1,), add_batch_rank=True),
                q_values_next_sampled=Tuple(FloatBox(shape=(1,)), FloatBox(shape=(1,)), add_batch_rank=True),
                q_values=Tuple(FloatBox(shape=(1,)), FloatBox(shape=(1,)), add_batch_rank=True),
                log_probs_sampled=FloatBox(shape=(1,), add_batch_rank=True),
                q_values_sampled=Tuple(FloatBox(shape=(1,)), FloatBox(shape=(1,)), add_batch_rank=True),
                rewards=FloatBox(add_batch_rank=True),
                terminals=BoolBox(add_batch_rank=True),
                loss_per_item=FloatBox(add_batch_rank=True)
            ),
            action_space=IntBox(2, shape=(), add_batch_rank=True)
        )
        return test

    def test_sac_loss_function(self):
        loss_function = SACLossFunction(
            target_entropy=0.1, discount=0.8
        )
        test = self._prepare_loss_function_test(loss_function)

        batch_size = 10
        inputs = [
            0.9,  # alpha
            [[0.5]] * batch_size,  # log_probs_next_sampled
            ([[0.0]] * batch_size, [[1.0]] * batch_size),  # q_values_next_sampled
            ([[0.0]] * batch_size, [[1.0]] * batch_size),  # q_values
            [[0.5]] * batch_size,  # log_probs_sampled
            ([[.1]] * batch_size, [[.2]] * batch_size),  # q_values_sampled
            [1.0] * batch_size,  # rewards
            [False] * batch_size  # terminals
        ]

        policy_loss_per_item = [.35] * batch_size
        values_loss_per_item = [.2696] * batch_size
        alpha_loss_per_item = [-.54] * batch_size

        test.test(
            (loss_function.loss, inputs),
            expected_outputs=[
                np.mean(policy_loss_per_item),
                policy_loss_per_item,
                np.mean(values_loss_per_item),
                values_loss_per_item,
                np.mean(alpha_loss_per_item),
                alpha_loss_per_item
            ],
            decimals=5
        )

        test.test(
            (loss_function.loss_per_item, inputs),
            expected_outputs=[
                policy_loss_per_item,
                values_loss_per_item,
                alpha_loss_per_item
            ],
            decimals=5
        )

        test.test(
            (loss_function.loss_average, [policy_loss_per_item]),
            expected_outputs=[np.mean(policy_loss_per_item)],
            decimals=5
        )

    def test_sac_loss_function_no_target_entropy(self):
        loss_function = SACLossFunction(
            target_entropy=None, discount=0.8
        )
        test = self._prepare_loss_function_test(loss_function)

        batch_size = 10
        inputs = [
            0.9,  # alpha
            [[0.5]] * batch_size,  # log_probs_next_sampled
            ([[0.0]] * batch_size, [[1.0]] * batch_size),  # q_values_next_sampled
            ([[0.0]] * batch_size, [[1.0]] * batch_size),  # q_values
            [[0.5]] * batch_size,  # log_probs_sampled
            ([[.1]] * batch_size, [[.2]] * batch_size),  # q_values_sampled
            [1.0] * batch_size,  # rewards
            [False] * batch_size  # terminals
        ]

        policy_loss_per_item = [.35] * batch_size
        values_loss_per_item = [.2696] * batch_size
        alpha_loss_per_item = [0.0] * batch_size

        test.test(
            (loss_function.loss, inputs),
            expected_outputs=[
                np.mean(policy_loss_per_item),
                policy_loss_per_item,
                np.mean(values_loss_per_item),
                values_loss_per_item,
                np.mean(alpha_loss_per_item),
                alpha_loss_per_item
            ],
            decimals=5
        )

        test.test(
            (loss_function.loss_per_item, inputs),
            expected_outputs=[
                policy_loss_per_item,
                values_loss_per_item,
                alpha_loss_per_item
            ],
            decimals=5
        )

        test.test(
            (loss_function.loss_average, [policy_loss_per_item]),
            expected_outputs=[np.mean(policy_loss_per_item)],
            decimals=5
        )
