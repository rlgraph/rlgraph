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

from yarl.components.loss_functions import DQNLossFunction
from yarl.spaces import *
from yarl.tests import ComponentTest

import numpy as np


class TestLossFunctions(unittest.TestCase):

    def test_dqn_loss_function_on_int_action_space(self):
        # Create a shape=() 2-action discrete-space.
        # Thus, each action pick consists of one single binary action (0 or 1).
        action_space = IntBox(2, shape=(), add_batch_rank=True)
        q_values_space = FloatBox(shape=action_space.get_shape(with_category_rank=True), add_batch_rank=True)
        dqn_loss_function = DQNLossFunction(discount=1.0)  # gamma=1.0: keep it simple

        test = ComponentTest(
            component=dqn_loss_function,
            input_spaces=dict(q_values=q_values_space, actions=action_space, rewards=FloatBox(add_batch_rank=True),
                              qt_values_s_=q_values_space
                              ),
            action_space=action_space
        )

        # Batch of size=2.
        input_ = dict(
            q_values=np.array([[10.0, -10.0], [-0.101, -90.6]]),
            actions=np.array([0, 1]),
            rewards=np.array([9.4, -1.23]),
            qt_values_s_=np.array([[12.0, -8.0], [22.3, 10.5]]),
        )
        """
        Calculation:
        batch of 2, gamma=1.0
        Qt(s'a') = [12 -8] [22.3 10.5] -> max(a') = [12] [22.3]
        Q(s,a)  = [10.0] [-90.6]
        L = E(batch)| ((r + gamma max(a')Qt(s'a') ) - Q(s,a))^2 |
        L = ((9.4 + 1.0*12 - 10.0)^2 + (-1.23 + 1.0*22.3 - -90.6)^2) / 2
        L = ((129.96) + (12470.1889)) / 2
        L = 6300.07445
        """

        # Batch size=2 -> Expect 2 values in the `loss_per_item` out-Socket.
        expected_loss_per_item = np.array([129.95999, 12470.188], dtype=np.float32)
        test.test(out_socket_names="loss_per_item", inputs=input_, expected_outputs=expected_loss_per_item)
        # Expect the mean over the batch.
        expected_loss = expected_loss_per_item.mean()
        test.test(out_socket_names="loss", inputs=input_, expected_outputs=expected_loss)

    def test_double_dqn_loss_function_on_int_action_space(self):
        # Create a shape=() 3-action discrete-space.
        # Thus, each action pick consists of one single action (0, 1, or 2).
        action_space = IntBox(3, shape=(), add_batch_rank=True)
        q_values_space = FloatBox(shape=action_space.get_shape(with_category_rank=True), add_batch_rank=True)
        dqn_loss_function = DQNLossFunction(double_q=True, discount=0.9)

        test = ComponentTest(
            component=dqn_loss_function,
            input_spaces=dict(q_values=q_values_space, actions=action_space,
                              rewards=FloatBox(add_batch_rank=True),
                              qt_values_s_=q_values_space,
                              q_values_s_=q_values_space,  # The Q-values for s' from the policy (not the target-policy)
                              ),
            action_space=action_space
        )

        # Batch of size=2.
        input_ = dict(
            q_values=np.array([[10.0, -10.0, 12.4], [-0.101, -4.6, -9.3]]),
            actions=np.array([2, 1]),
            rewards=np.array([10.3, -4.25]),
            qt_values_s_=np.array([[-12.3, 1.2, 1.4], [12.2, -11.5, 9.2]]),
            q_values_s_=np.array([[-10.3, 1.5, 1.4], [8.2, -10.9, 9.3]]),
        )
        """
        Calculation:
        batch of 2, gamma=0.9
        a' = [1 2]  <- argmax(a')Q(s'a')
        Qt(s'.) = [-12.3  1.2  1.4] [12.2  -11.5  9.2] -> Qt(s'a') = [1.2] [9.2]
        a = [2 1]
        Q(s,a)  = [12.4] [-4.6]
        L = E(batch)| ((r + gamma Qt(s'( argmax(a') Q(s'a') )) ) - Q(s,a))^2 |
        L = ((10.3 + 0.9*1.2 - 12.4)^2 + (-4.25 + 0.9*9.2 - -4.6)^2) / 2
        L = ((1.0404) + (74.4769)) / 2
        L = 37.75865
        """

        # Batch size=2 -> Expect 2 values in the `loss_per_item` out-Socket.
        expected_loss_per_item = np.array([1.040399, 74.47688], dtype=np.float32)
        test.test(out_socket_names="loss_per_item", inputs=input_, expected_outputs=expected_loss_per_item)
        # Expect the mean over the batch.
        expected_loss = expected_loss_per_item.mean()
        test.test(out_socket_names="loss", inputs=input_, expected_outputs=expected_loss)

    def test_dqn_loss_function_in_multi_action_space(self):
        # Create a shape=(2,) 4-action discrete-space.
        # Thus, each action pick consists of 2 composite-actions chosen from a set of 4 possible single actions.
        action_space = IntBox(4, shape=(2,), add_batch_rank=True)
        q_values_space = FloatBox(shape=action_space.get_shape(with_category_rank=True), add_batch_rank=True)
        dqn_loss_function = DQNLossFunction(discount=1.0)  # gamma=1.0: keep it simple

        test = ComponentTest(
            component=dqn_loss_function,
            input_spaces=dict(q_values=q_values_space, actions=action_space, rewards=FloatBox(add_batch_rank=True),
                              qt_values_s_=q_values_space
                              ),
            action_space=action_space
        )

        # Batch of size=2.
        input_ = dict(
            q_values=np.array([[[10.0, -10.0, 9.8, 2.0], [20.2, -0.6, 0.001, 98.1]],
                               [[4.1, -11.1, 7.5, 2.1], [21.3, 9.5, -0.101, -90.6]]]),
            actions=np.array([[0, 3], [1, 2]]),
            rewards=np.array([9.4, -1.23]),
            qt_values_s_=np.array([[[12.0, -8.0, 7.8, 4.0], [16.2, -2.6, -6.001, 90.1]],
                               [[5.1, -12.1, 8.5, 3.1], [22.3, 10.5, 1.098, -89.2]]]),
        )
        # Batch size=2 -> Expect 2 values in the `loss_per_item` out-Socket.
        expected_loss_per_item = np.array([1.177221, 629.2822], dtype=np.float32)
        test.test(out_socket_names="loss_per_item", inputs=input_, expected_outputs=expected_loss_per_item)
        # Just expect the mean over the batch.
        expected_loss = expected_loss_per_item.mean()
        test.test(out_socket_names="loss", inputs=input_, expected_outputs=expected_loss)

