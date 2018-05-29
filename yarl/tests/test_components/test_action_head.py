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

from yarl.components.action_heads import ActionHead
from yarl.spaces import *
from yarl.tests import ComponentTest

import numpy as np


class TestActionHead(unittest.TestCase):

    def test_action_head_with_discrete_action_space(self):

        # 2x2 action-pick, each action with 5 categories.
        space = IntBox(5, shape=(2, 2), add_batch_rank=True)
        # Our NN must output this Space then:
        nn_output_space = FloatBox(shape=(space.flat_dim_with_categories,), add_batch_rank=True)

        # The Component to test.
        action_head = ActionHead(action_space=space)
        test = ComponentTest(component=action_head, input_spaces=dict(nn_output=nn_output_space,
                                                                      time_step=int))

        # NN-output (batch-size=2).
        inputs = np.array([[[0.5, 0.25, 0.25],
                                     [0.98, 0.01, 0.01],
                                     [0.0, 0.6, 0.4],
                                     [0.2, 0.25, 0.55],
                                     [0.3, 0.3, 0.4]
                                     ]])
        expected = np.array([[0, 0, 1, 2, 2]])
        test.test(out_socket_name="draw", inputs=inputs, expected_outputs=expected)

