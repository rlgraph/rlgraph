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
        # state_space (NN is a simple single fc-layer relu network (2 units), random biases, random weights).
        state_space = FloatBox(shape=(4,), add_batch_rank=True)

        # action_space (5 possible actions).
        action_space = IntBox(5, add_batch_rank=True)

        policy = Policy(neural_network="configs/test_simple_nn.json")
        test = ComponentTest(component=policy, input_spaces=dict(nn_input=state_space), action_space=action_space)

        # Some NN inputs (4 input nodes, batch size=2).
        states = np.array([[-0.08, 0.4, -0.05, -0.55], [13.0, -14.0, 10.0, -16.0]])
        # Raw NN-output.
        expected_nn_output = np.array([[-0.1824044, -0.2356079], [0.2732344, -7.3936715]], dtype=np.float32)
        test.test(out_socket_names="nn_output", inputs=states, expected_outputs=expected_nn_output)

        # Raw action layer output; Expected shape=(2,5): 2=batch, 5=action categories
        expected_action_layer_output = np.array(
            [
                [-0.1187086, -0.06067234, 0.12123819, -0.14454277, -0.06990821],
                [-4.9992857, 3.4695446, 4.8772826, -4.974872, 2.348141]
            ], dtype=np.float32)
        test.test(out_socket_names="action_layer_output", inputs=states, expected_outputs=expected_action_layer_output)

        # Parameter (probabilities). Softmaxed action_layer_outputs.
        expected_probabilities_output = np.array(
            [
                [0.18672596, 0.19788347, 0.23736258, 0.18196383, 0.19606426],
                [3.8779312e-05, 1.8474223e-01, 7.5498617e-01, 3.9737686e-05, 6.0193103e-02]
            ], dtype=np.float32)
        test.test(out_socket_names="parameters", inputs=states, expected_outputs=expected_probabilities_output)
        # Logits: log of the parameters.
        test.test(out_socket_names="logits", inputs=states, expected_outputs=np.log(expected_probabilities_output))

        # Stochastic sample.
        expected_actions = np.array([2, 2])
        test.test(out_socket_names="sample_stochastic", inputs=states, expected_outputs=expected_actions)

        # Deterministic sample.
        expected_actions = np.array([2, 2])
        test.test(out_socket_names="sample_deterministic", inputs=states, expected_outputs=expected_actions)

        # Distribution's entropy.
        expected_h = np.array([1.6048075, 0.694136])
        test.test(out_socket_names="entropy", inputs=states, expected_outputs=expected_h)
