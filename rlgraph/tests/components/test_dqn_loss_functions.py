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

import numpy as np
from rlgraph.components.loss_functions import DQNLossFunction
from rlgraph.spaces import *
from rlgraph.tests import ComponentTest


class TestDQNLossFunctions(unittest.TestCase):

    reward_space = FloatBox(add_batch_rank=True)
    terminal_space = BoolBox(add_batch_rank=True)
    loss_per_item_space = FloatBox(add_batch_rank=True)

    def test_dqn_loss_function_on_int_action_space(self):
        # Create a shape=() 2-action discrete-space.
        # Thus, each action pick consists of one single binary action (0 or 1).
        action_space = IntBox(2, shape=(), add_batch_rank=True)
        q_values_space = FloatBox(shape=action_space.get_shape(with_category_rank=True), add_batch_rank=True)
        dqn_loss_function = DQNLossFunction(discount=1.0)  # gamma=1.0: keep it simple

        test = ComponentTest(
            component=dqn_loss_function,
            input_spaces=dict(
                q_values_s=q_values_space,
                actions=action_space,
                rewards=self.reward_space,
                terminals=self.terminal_space,
                qt_values_sp=q_values_space,
                loss_per_item=self.loss_per_item_space
            ),
            action_space=action_space
        )

        # Batch of size=2.
        input_ = [
            np.array([[10.0, -10.0], [-0.101, -90.6]]),
            np.array([0, 1]),
            np.array([9.4, -1.23]),
            np.array([False, False]),
            np.array([[12.0, -8.0], [22.3, 10.5]])
        ]
        """
        Calculation:
        batch of 2, gamma=1.0
        Qt(s'a') = [12 -8] [22.3 10.5] -> max(a') = [12] [22.3]
        Q(s,a)  = [10.0] [-90.6]
        L = E(batch)| 0.5((r + gamma max(a')Qt(s'a') ) - Q(s,a))^2 |
        L = (0.5(9.4 + 1.0*12 - 10.0)^2 + 0.5(-1.23 + 1.0*22.3 - -90.6)^2) / 2
        L = (0.5(129.96) + 0.5(12470.1889)) / 2
        L = (64.98 + 6235.09445) / 2
        L = 3150.037225
        """

        # Batch size=2 -> Expect 2 values returned by `loss_per_item`.
        expected_loss_per_item = np.array([64.979996, 6235.09445], dtype=np.float32)
        test.test(("loss_per_item", input_), expected_outputs=expected_loss_per_item)
        # Expect the mean over the batch.
        expected_loss = expected_loss_per_item.mean()
        test.test(("loss_average", expected_loss_per_item), expected_outputs=expected_loss)
        # Both.
        test.test(("loss", input_), expected_outputs=[expected_loss, expected_loss_per_item])

    def test_double_dqn_loss_function_on_int_action_space(self):
        # Create a shape=() 3-action discrete-space.
        # Thus, each action pick consists of one single action (0, 1, or 2).
        action_space = IntBox(3, shape=(), add_batch_rank=True)
        q_values_space = FloatBox(shape=action_space.get_shape(with_category_rank=True), add_batch_rank=True)
        dqn_loss_function = DQNLossFunction(double_q=True, discount=0.9)

        test = ComponentTest(
            component=dqn_loss_function,
            input_spaces=dict(
                q_values_s=q_values_space,
                actions=action_space,
                rewards=self.reward_space,
                terminals=self.terminal_space,
                qt_values_sp=q_values_space,
                q_values_sp=q_values_space,
                loss_per_item=self.loss_per_item_space
            ),
            action_space=action_space
        )

        # Batch of size=2.
        input_ = [
            np.array([[10.0, -10.0, 12.4], [-0.101, -4.6, -9.3]]),
            np.array([2, 1]),
            np.array([10.3, -4.25]),
            np.array([False, True]),
            np.array([[-12.3, 1.2, 1.4], [12.2, -11.5, 9.2]]),
            np.array([[-10.3, 1.5, 1.4], [8.2, -10.9, 9.3]])
        ]
        """
        Calculation:
        batch of 2, gamma=0.9
        a' = [1 2]  <- argmax(a')Q(s'a')
        Qt(s'.) = [-12.3  1.2  1.4] [12.2  -11.5  9.2] -> Qt(s'a') = [1.2] [0.0 <- normally 9.2, but terminal(!) = True]
        a = [2 1]
        Q(s,a)  = [12.4] [-4.6]
        L = E(batch)| 0.5((r + gamma Qt(s'( argmax(a') Q(s'a') )) ) - Q(s,a))^2 |
        L = (0.5(10.3 + 0.9*1.2 - 12.4)^2 + 0.5(-4.25 + 0.9*0.0 - -4.6)^2) / 2
        L = (0.5(1.0404) + 0.5(0.1224999)) / 2
        L = (0.5202 + 0.06124995) / 2 
        L = 0.290725
        """

        # Batch size=2 -> Expect 2 values returned by `loss_per_item`.
        expected_loss_per_item = np.array([0.5202, 0.06125], dtype=np.float32)
        test.test(("loss_per_item", input_), expected_outputs=expected_loss_per_item, decimals=4)
        # Expect the mean over the batch.
        expected_loss = expected_loss_per_item.mean()
        test.test(("loss_average", expected_loss_per_item), expected_outputs=expected_loss, decimals=4)
        # Both.
        test.test(("loss", input_), expected_outputs=[expected_loss, expected_loss_per_item], decimals=4)

    def test_dqn_loss_function_in_multi_action_space(self):
        # Create a shape=(3,) 4-action discrete-space.
        # Thus, each action pick consists of 2 composite-actions chosen from a set of 4 possible single actions.
        action_space = IntBox(4, shape=(3,), add_batch_rank=True)
        q_values_space = FloatBox(shape=action_space.get_shape(with_category_rank=True), add_batch_rank=True)
        dqn_loss_function = DQNLossFunction(discount=0.8)

        test = ComponentTest(
            component=dqn_loss_function,
            input_spaces=dict(
                q_values_s=q_values_space,
                actions=action_space,
                rewards=self.reward_space,
                terminals=self.terminal_space,
                qt_values_sp=q_values_space,
                loss_per_item=self.loss_per_item_space
            ),
            action_space=action_space
        )

        # Batch of size=2.
        input_ = [
            np.array([[[10.0, -10.0, 9.8, 2.0], [20.2, -0.6, 0.001, 98.1], [10.0, -10.0, 9.8, 2.0]],
                               [[4.1, -11.1, 7.5, 2.1], [21.3, 9.5, -0.101, -90.6], [21.3, 9.5, -0.101, -90.6]]]),
            np.array([[0, 3, 2], [1, 2, 0]]),
            np.array([9.4, -1.23]),
            np.array([False, True]),
            np.array([[[12.0, -8.0, 7.8, 4.0], [16.2, -2.6, -6.001, 90.1], [12.0, -8.0, 7.8, 4.0]],
                                   [[5.1, -12.1, 8.5, 3.1], [22.3, 10.5, 1.098, -89.2], [22.3, 10.5, 1.098, -89.2]]])
        ]

        """
        Calculation:
        batch of 2, gamma=0.8
        Qt(s'a') = [[12 -8 7.8 4] [16.2 -2.6 -6.001 90.1] .. ] [[5.1 -12.1 8.5 3.1] [22.3 10.5 1.098 -89.2] ..] ->
            max(a')Qt(s'a') = [[12 90.1 12] [0 0 0] <- would have been [8.5 22.3], but terminal=True]
        Q(s,a)  = [10.0 98.1 9.8] [-11.1 -0.101 21.3]
        L = E(batch)| 0.5((r + gamma max(a')Qt(s'a') ) - Q(s,a))^2 |
        L = (0.5((9.4 + 0.8*12 - 10.0) + (9.4 + 0.8*90.1 - 98.1) + (9.4 + 0.8*12 - 9.8))/3)^2 +
            (0.5(((-1.23 + 0.8*0.0 - -11.1) + (-1.23 + 0.8*0.0 - -0.101) + (-1.23 + 0.8*0.0 - 21.3))/3)^2) / 2
        L = (0.5((9.0) + (-16.62) + (9.2))/3)^2 +
            (0.5(((-1.23 + 0.8*0.0 - -11.1) + (-1.23 + 0.8*0.0 - -0.101) + (-1.23 + 0.8*0.0 - 21.3))/3)^2) / 2
        L = (0.5(0.5267)^2 + 0.5(-4.59633)^2) / 2
        L = (0.138689725 + 10.563136) / 2
        L = 5.3509128625
        """

        # Batch size=2 -> Expect 2 values returned by `loss_per_item`.
        expected_loss_per_item = np.array([0.138689725, 10.563136], dtype=np.float32)
        print(test.test(("loss_per_item", input_), expected_outputs=None))
        # Just expect the mean over the batch.
        expected_loss = expected_loss_per_item.mean()
        test.test(("loss_average", expected_loss_per_item), expected_outputs=expected_loss)
        # Both.
        test.test(("loss", input_), expected_outputs=[expected_loss, expected_loss_per_item])

    def test_double_dqn_loss_function_on_multi_int_action_space(self):
        # Create a shape=(2,2) 3-action discrete-space.
        # Thus, each action pick consists of one single action (0, 1, or 2).
        action_space = IntBox(3, shape=(2,2), add_batch_rank=True)
        q_values_space = FloatBox(shape=action_space.get_shape(with_category_rank=True), add_batch_rank=True)
        dqn_loss_function = DQNLossFunction(double_q=True, discount=1.0)

        test = ComponentTest(
            component=dqn_loss_function,
            input_spaces=dict(
                q_values_s=q_values_space,
                actions=action_space,
                rewards=self.reward_space,
                terminals=self.terminal_space,
                qt_values_sp=q_values_space,
                q_values_sp=q_values_space,
                loss_per_item=self.loss_per_item_space
            ),
            action_space=action_space
        )

        # Batch of size=4.
        input_ = [
            np.array(
                [
                    [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10., 11., 12.]]],
                    [[[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], [[8.0, 9.0, 10.], [11., 12., 13.]]],
                    [[[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]], [[9.0, 10., 11.], [12., 13., 14.]]],
                    [[[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], [[10., 11., 12.], [13., 14., 15.]]],
                ]
            ),
            np.array(
                [
                    [[0, 1], [2, 0]],
                    [[1, 2], [0, 1]],
                    [[2, 0], [1, 2]],
                    [[0, 1], [1, 1]]
                ]
            ),
            np.array([-1.0, -2.0, -3.0, -4.0]),
            np.array([False, False, True, False]),
            np.array(
                [
                    [[[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], [[7.5, 8.5, 9.5], [10.5, 11.5, 12.5]]],
                    [[[2.5, 3.5, 4.5], [5.5, 6.5, 7.5]], [[8.5, 9.5, 10.5], [11.5, 12.5, 13.5]]],
                    [[[3.5, 4.5, 5.5], [6.5, 7.5, 8.5]], [[9.5, 10.5, 11.5], [12.5, 13.5, 14.5]]],
                    [[[4.5, 5.5, 6.5], [7.5, 8.5, 9.5]], [[10.5, 11.5, 12.5], [13.5, 14.5, 15.5]]],
                ]
            ),
            np.array(
                [
                    [[[1.6, 2.6, 3.6], [4.6, 5.6, 6.6]], [[7.6, 8.6, 9.6], [10.6, 11.6, 12.6]]],
                    [[[2.6, 3.6, 4.6], [5.6, 6.6, 7.6]], [[8.6, 9.6, 10.6], [11.6, 12.6, 13.6]]],
                    [[[3.6, 4.6, 5.6], [6.6, 7.6, 8.6]], [[9.6, 10.6, 11.6], [12.6, 13.6, 14.6]]],
                    [[[4.6, 5.6, 6.6], [7.6, 8.6, 9.6]], [[10.6, 11.6, 12.6], [13.6, 14.6, 15.6]]],
                ]
            )
        ]
        """
        Calculation:
        batch of 4, gamma=1.0
        a' = [[2 2] [2 2]], [[2 2] [2 2]], [[2 2] [2 2]], [[2 2] [2 2]]  <- argmax(a')Q(s'a')
        Qt(s'a') = [[3.5 6.5] [9.5 12.5]], [[4.5 7.5] [10.5 13.5]], [[0.0 0.0] [0.0 0.0]], [[6.5 9.5] [12.5 15.5]]
        a = [[0 1] [2 0]], [[1 2] [0 1]], [[2 0] [1 2]], [[0 1] [1 1]]
        Q(s,a) = [[1 5] [9 10]], [[3 7] [8 12]], [[5 6] [10 14]], [[4 8] [11 14]]
        L = E(batch)| 0.5((r + gamma Qt(s'( argmax(a') Q(s'a') )) ) - Q(s,a))^2 |
        L = [
            0.5(((-1 + 3.5 - 1) + (-1 + 6.5 - 5) + (-1 + 9.5 - 9) + (-1 + 12.5 - 10))/4)^2 +   -> 0.28125
            0.5(((-2 + 4.5 - 3) + (-2 + 7.5 - 7) + (-2 + 10.5 - 8) + (-2 + 13.5 - 12))/4)^2 +  -> 0.125
            0.5(((-3 + 0.0 - 5) + (-3 + 0.0 - 6) + (-3 + 0.0 - 10) + (-3 + 0.0 - 14))/4)^2 +   -> 69.03125
            0.5(((-4 + 6.5 - 4) + (-4 + 9.5 - 8) + (-4 + 12.5 - 11) + (-4 + 15.5 - 14))/4)^2   -> 2.53125
            ] / 4
        L = 17.9921875
        """

        # Batch size=2 -> Expect 2 values returned by `loss_per_item`.
        expected_loss_per_item = np.array([0.28125, 0.125, 69.03125, 2.53125], dtype=np.float32)
        test.test(("loss_per_item", input_), expected_outputs=expected_loss_per_item)
        # Expect the mean over the batch.
        expected_loss = expected_loss_per_item.mean()
        test.test(("loss_average", expected_loss_per_item), expected_outputs=expected_loss)
        # Both.
        test.test(("loss", input_), expected_outputs=[expected_loss, expected_loss_per_item])

    def test_dqn_loss_function_in_container_action_space(self):
        action_space = Dict({"a": IntBox(2), "b": {"ba": IntBox(3), "bb": IntBox(2)}}, add_batch_rank=True)
        q_values_space = Dict({"a": FloatBox(shape=(2,)), "b": {"ba": FloatBox(shape=(3,)),
                                                                "bb": FloatBox(shape=(2, ))}}, add_batch_rank=True)
        dqn_loss_function = DQNLossFunction(discount=1.0)

        test = ComponentTest(
            component=dqn_loss_function,
            input_spaces=dict(
                q_values_s=q_values_space,
                actions=action_space,
                rewards=self.reward_space,
                terminals=self.terminal_space,
                qt_values_sp=q_values_space,
                loss_per_item=self.loss_per_item_space
            ),
            action_space=action_space
        )

        # Batch of size=1.
        input_ = [
            # q(s)-values
            {"a": np.array([[1.0, 2.0]]), "b": {"ba": np.array([[0.0, -0.5, 1.2]]), "bb": np.array([[-1.0, -2.0]])}},
            # actions
            {"a": np.array([0]), "b": {"ba": np.array([0]), "bb": np.array([1])}},
            np.array([1.0]),
            np.array([False]),
            # qt(s')-values
            {"a": np.array([[-1.0, -2.0]]), "b": {"ba": np.array([[3.0, 3.1, 3.2]]), "bb": np.array([[1.0, -5.2]])}}
        ]

        """
        Calculation:
        batch of 1, gamma=1.0
        TDtarget (global across all sub-actions) = 1/N SUMd ( r + gamma max(a')Qdt(s'a') )
          = 1/3 [(1.0 + 1.0*-1.0) + (1.0 + 1.0*3.2) + (1.0 + 1.0*1.0)] = 2.06667
        L = E(batch) | 1/N SUMd ( 0.5*(TDtarget - Qd(s,a))^2 ) |   # d=action components
        L = 1/3 SUMd ( 0.5[a]^2 + 0.5[ba]^2 + 0.5[bb]^2 )
            a=2.06667 - 1.0=1.06667
            ba=2.06667 - 0.0= 2.06667
            bb=2.06667 - -2.0=4.06667
            
            Huberloss/square before aggregation:
            SUM=0.5(1.06667)^2 + 0.5(2.06667)^2 + 0.5(4.06667)^2 = 10.973357
            10.973357/3 = 3.657786

            Huber loss/square after aggregation: 
            SUM=0.5 * ((1.06667 + 2.06667 + 4.06667)/3) ^ 2
               =2.88
        """

        # Batch size=2 -> Expect 2 values returned by `loss_per_item`.
        expected_loss_per_item = np.array([2.8800], dtype=np.float32)
        test.test(("loss_per_item", input_), expected_outputs=expected_loss_per_item, decimals=4)
        # Just expect the mean over the batch.
        expected_loss = expected_loss_per_item.mean()
        test.test(("loss_average", expected_loss_per_item), expected_outputs=expected_loss, decimals=4)
        # Both.
        test.test(("loss", input_), expected_outputs=[expected_loss, expected_loss_per_item], decimals=2)

    def test_double_dqn_loss_function_in_container_action_space(self):
        action_space = Dict({"a": IntBox(2), "b": {"ba": IntBox(3), "bb": IntBox(2)}}, add_batch_rank=True)
        q_values_space = Dict({"a": FloatBox(shape=(2,)), "b": {"ba": FloatBox(shape=(3,)),
                                                                "bb": FloatBox(shape=(2, ))}}, add_batch_rank=True)
        dqn_loss_function = DQNLossFunction(discount=0.99, double_q=True)

        test = ComponentTest(
            component=dqn_loss_function,
            input_spaces=dict(
                q_values_s=q_values_space,
                actions=action_space,
                rewards=self.reward_space,
                terminals=self.terminal_space,
                qt_values_sp=q_values_space,
                q_values_sp=q_values_space,
                loss_per_item=self.loss_per_item_space
            ),
            action_space=action_space
        )

        # Batch of size=2.
        input_ = [
            # q(s)-values
            {"a": np.array([[1.0, 2.0], [3.0, 4.0]]), "b": {"ba": np.array([[0.0, -0.5, 1.2], [-0.1, -0.2, -0.3]]),
                                                            "bb": np.array([[-1.0, -2.0], [0.5, 0.6]])}},
            # actions
            {"a": np.array([0, 1]), "b": {"ba": np.array([0, 2]), "bb": np.array([1, 0])}},
            np.array([1.0, -1.0]),
            np.array([False, True]),
            # qt(s')-values
            {"a": np.array([[-1.0, -2.0], [5.0, 6.0]]), "b": {"ba": np.array([[3.0, 3.1, 3.2], [4.1, 4.2, 4.3]]),
                                                              "bb": np.array([[1.0, -5.2], [-0.5, -0.6]])}},
            # q(s')-values
            {"a": np.array([[-1.0, 100.0], [5.0, 60.0]]), "b": {"ba": np.array([[3.0, 3.1, 3.2], [4.5, 4.4, 4.3]]),
                                                                "bb": np.array([[1.0, -5.2], [-0.5, 0.6]])}}
        ]

        """
        Calculation:
        batch of 2, gamma=0.99
        argmaxa'Qd(s',a'): a=[1, 1] ba=[2, 0] bb=[0, 1]
        TDtarget (global across all sub-actions) = 1/N SUMd ( r + gamma Qdt(s',argmaxa'Qd(s',a')) )
          = [
                1/3 [(1.0 + 0.99*-2.0) + (1.0 + 0.99*3.2) + (1.0 + 0.99*1.0)],
                1/3 [(-1.0 + 0.99*0.0) + (-1.0 + 0.99*0.1) + (-1.0 + 0.99*0.0)]  # all 0 due to terminal=True
            ]
          = [1/3(-0.98 + 4.168 + 1.99), 1/3(-1.0 + -1.0 + -1.0)]
          = [1.726, -1.0]
        L = E(batch) | 1/N SUMd ( 0.5*(TDtarget - Qd(s,a))^2 ) |   # d=action components
        L = 1/3 SUMd ( 0.5[a]^2 + 0.5[ba]^2 + 0.5[bb]^2 )
            a=[1.726-1.0, -1.0-4.0]=[0.726, -5.0]
            ba=[1.726-0.0, -1.0+0.3]=[1.726, -0.7]
            bb=[1.726+2.0, -1.0-0.5]=[3.726, -1.5]
            
            Huberloss/square before aggregation:
            SUM=[
                0.5(0.726)^2 + 0.5(1.726)^2 + 0.5(3.726)^2,
                0.5(-5.0)^2 + 0.5(-0.7)^2 + 0.5(-1.5)^2
                ] = [8.694614, 13.87]
            /3 = [2.89820467, 4.62333]
            
            Huberloss/square after aggregation:
             =[
                0.5 * ((0.726 + 1.726 + 3.726) / 3)^2,
                0.5 * ((-5.0 + -0.7 + -1.5) / 3)^2
                ] = [2.1204 2.88]
            
        """

        # Batch size=2 -> Expect 2 values returned by `loss_per_item`.
        expected_loss_per_item = np.array([2.1204, 2.88], dtype=np.float32)
        test.test(("loss_per_item", input_), expected_outputs=expected_loss_per_item, decimals=4, print=True)
        # Just expect the mean over the batch.
        expected_loss = expected_loss_per_item.mean()
        test.test(("loss_average", expected_loss_per_item), expected_outputs=expected_loss, decimals=4)
        # Both.
        test.test(("loss", input_), expected_outputs=[expected_loss, expected_loss_per_item], decimals=2)

