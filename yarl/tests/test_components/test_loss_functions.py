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

    def test_dqn_loss_function(self):
        # Create a shape=(2,) 4-action discrete-space.
        # Thus, each action pick consists of 2 composite-actions chosen from a set of 4 possible single actions.
        action_space = IntBox(4, shape=(2,), add_batch_rank=True)
        q_values_space = FloatBox(shape=action_space.shape, add_batch_rank=True)

        dqn_loss_function = DQNLossFunction(discount=1.0)  # gamma=1.0: keep it simple
        test = ComponentTest(component=dqn_loss_function,
                             input_spaces=dict(q_values=q_values_space,
                                               actions=action_space,
                                               rewards=FloatBox(add_batch_rank=True),
                                               q_values_s_=q_values_space))

        # Batch of size=2.
        input_ = dict(
            q_values=np.array([[[10.0, -10.0, 9.8, 2.0], [20.2, -0.6, 0.001, 98.1]],
                               [[4.1, -11.1, 7.5, 2.1], [21.3, 9.5, -0.101, -90.6]]]),
            actions=np.array([[0, 3], [1, 2]]),
            rewards=np.array([9.4, -1.23]),
            q_values_s_=np.array([[[12.0, -8.0, 7.8, 4.0], [16.2, -2.6, -6.001, 90.1]],
                               [[5.1, -12.1, 8.5, 3.1], [22.3, 10.5, 1.098, -89.2]]]),
        )
        expected_loss_per_item = np.array([100, 100])
        expected_loss = 50.0
        test.test(out_socket_name="loss_per_item", inputs=input_, expected_outputs=expected_loss_per_item)
        test.test(out_socket_name="loss", inputs=input_, expected_outputs=expected_loss)

