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
        # state_space (NN is a simple single fc-layer network (2 units) with relu activation and 4 input nodes).
        state_space = FloatBox(shape=(4,), add_batch_rank=True)

        # action_space (5 possible actions).
        action_space = IntBox(5, add_batch_rank=True)

        policy = Policy(neural_network="configs/test_simple_nn.json")
        test = ComponentTest(component=policy, input_spaces=dict(nn_input=state_space), action_space=action_space)

        # Some NN inputs (4 input nodes, batch size=4).
        states = np.array([[-0.08, 0.4, -0.05, -0.55], [13.0, -14.0, 10.0, -16.0],
                           [1.75, 1.875, -1.9876, 0.987234], [25.0, -26.0, 27.0, 128.0]])
        # Raw NN-output.
        expected_nn_output = np.array([[-0.1824044, -0.2356079], [0.2732344, -7.393671],
                                       [-1.56982, -0.9464841], [64.854836, 64.18529]], dtype=np.float32)
        test.test(out_socket_names="nn_output", inputs=states, expected_outputs=expected_nn_output)

        # Raw logits output (from ActionLayer); Expected shape=(4,5): 4=batch, 5=action categories
        expected_action_layer_output = np.array(
            [
                [-0.1187086, -0.06067234, 0.12123819, -0.14454277, -0.06990821],
                [-4.999285, 3.4695446, 4.877282, -4.9748716, 2.3481407],
                [-0.29905114, -0.99373317, 0.337321, -0.519393, -0.91477036],
                [29.11778, 30.114904, -30.316084, 38.267143, 30.528452]
            ], dtype=np.float32)
        test.test(out_socket_names="logits", inputs=states, expected_outputs=expected_action_layer_output)

        # Stochastic sample.
        expected_actions = np.array([3, 2, 3, 3])
        test.test(out_socket_names="sample_stochastic", inputs=states, expected_outputs=expected_actions)

        # Deterministic sample.
        expected_actions = np.array([2, 2, 2, 3])
        test.test(out_socket_names="sample_deterministic", inputs=states, expected_outputs=expected_actions)

        # Distribution's entropy.
        expected_h = np.array([1.6048075, 0.694136, 1.4810212, 0.00753126])
        test.test(out_socket_names="entropy", inputs=states, expected_outputs=expected_h)
