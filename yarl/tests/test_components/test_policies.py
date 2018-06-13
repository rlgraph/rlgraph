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

import numpy as np
import unittest

from yarl.components.neural_networks import Policy
from yarl.spaces import *
from yarl.tests import ComponentTest


class TestPolicies(unittest.TestCase):

    def test_policy_for_discrete_action_space(self):
        # state_space (NN is a simple 2 fc-layer network with an arbitrary number of input nodes).
        state_space = FloatBox(shape=(4,), add_batch_rank=True)  # 4 input nodes

        # action_space (2 possible actions).
        action_space = IntBox(2, add_batch_rank=True)

        policy = Policy(neural_network="configs/test_simple_nn.json")
        test = ComponentTest(component=policy, input_spaces=dict(nn_input=state_space), action_space=action_space)

        # Some NN inputs.
        states = np.array([[-0.08, 0.4, -0.05, -0.55], [13.0, -14.0, 10.0, -16.0],
                           [1.75, 1.875, -1.9876, 0.987234], [25.0, -26.0, 27.0, 128.0]])
        # Raw NN-output.
        expected_nn_output = np.array([[0.0646258, -0.1633786], [1.9997894, 1.9039462],
                                       [0.26355803, -1.6376612], [-17.677073,  62.285240]], dtype=np.float32)
        test.test(out_socket_names="nn_output", inputs=states, expected_outputs=expected_nn_output)

        # Stochastic sample.
        expected_actions = np.array([0, 1, 0, 1])
        test.test(out_socket_names="sample_stochastic", inputs=states, expected_outputs=expected_actions)

        # Deterministic sample.
        expected_actions = np.array([0, 0, 0, 1])
        test.test(out_socket_names="sample_deterministic", inputs=states, expected_outputs=expected_actions)

        # Distribution's entropy.
        expected_h = np.array([0.68669093, 0.69200027, 0.38633075, 0.000014769186])
        test.test(out_socket_names="entropy", inputs=states, expected_outputs=expected_h)
