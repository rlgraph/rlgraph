# Copyright 2018 The RLgraph authors. All Rights Reserved.
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

from rlgraph.components.neural_networks import Policy
from rlgraph.spaces import *
from rlgraph.tests import ComponentTest
from rlgraph.utils import softmax, relu


class TestPolicies(unittest.TestCase):

    def test_policy_for_discrete_action_space(self):
        # state_space (NN is a simple single fc-layer relu network (2 units), random biases, random weights).
        state_space = FloatBox(shape=(4,), add_batch_rank=True)

        # action_space (5 possible actions).
        action_space = IntBox(5, add_batch_rank=True)

        policy = Policy(neural_network="configs/test_simple_nn.json", action_space=action_space)
        test = ComponentTest(
            component=policy,
            input_spaces=dict(
                get_nn_output=state_space,
                get_action_layer_output=state_space,
                get_entropy=state_space,
                get_logits_parameters_log_probs=state_space,
                get_q_values=state_space,
                sample_deterministic=state_space,
                sample_stochastic=state_space
            ),
            action_space=action_space
        )
        policy_params = test.graph_executor.read_variable_values(policy.variables)

        # Some NN api_methods (4 input nodes, batch size=2).
        states = np.array([[-0.08, 0.4, -0.05, -0.55], [13.0, -14.0, 10.0, -16.0]])
        # Raw NN-output.
        expected_nn_output = np.matmul(states, policy_params["policy/test-network/hidden-layer/dense/kernel"])
        test.test(("get_nn_output", states), expected_outputs=expected_nn_output, decimals=6)

        # Raw action layer output; Expected shape=(2,5): 2=batch, 5=action categories
        expected_action_layer_output = np.matmul(
            expected_nn_output, policy_params["policy/action-adapter/action-layer/dense/kernel"]
        )
        expected_action_layer_output = np.reshape(expected_action_layer_output, newshape=(2, 5))
        test.test(("get_action_layer_output", states), expected_outputs=expected_action_layer_output,
                  decimals=5)

        # Logits, parameters (probs) and skip log-probs (numerically unstable for small probs).
        expected_probabilities_output = softmax(expected_action_layer_output, axis=-1)
        test.test(("get_logits_parameters_log_probs", states, [0, 1]), expected_outputs=[
            expected_action_layer_output,
            np.array(expected_probabilities_output, dtype=np.float32)
            #np.log(expected_probabilities_output)
        ], decimals=5)

        # Stochastic sample.
        expected_actions = np.array([0, 4])
        test.test(("sample_stochastic", states), expected_outputs=expected_actions)

        # Deterministic sample.
        expected_actions = np.array([2, 4])
        test.test(("sample_deterministic", states), expected_outputs=expected_actions)

        # Distribution's entropy.
        expected_h = np.array([1.5822479, 0.0047392])
        test.test(("get_entropy", states), expected_outputs=expected_h)

    def test_policy_for_discrete_action_space_with_dueling_layer(self):
        # state_space (NN is a simple single fc-layer relu network (2 units), random biases, random weights).
        state_space = FloatBox(shape=(3,), add_batch_rank=True)

        # action_space (2 possible actions).
        action_space = IntBox(2, add_batch_rank=True)

        # Policy with additional dueling layer.
        policy = Policy(
            neural_network="configs/test_lrelu_nn.json",
            action_adapter_spec=dict(add_dueling_layer=True, action_space=action_space),
            switched_off_apis={"get_q_values"}
        )
        test = ComponentTest(
            component=policy,
            input_spaces=dict(
                get_nn_output=state_space,
                get_action_layer_output=state_space,
                get_entropy=state_space,
                get_dueling_output=state_space,
                get_logits_parameters_log_probs=state_space,
                #get_q_values=state_space,
                sample_deterministic=state_space,
                sample_stochastic=state_space
            ),
            action_space=action_space
        )
        policy_params = test.graph_executor.read_variable_values(policy.variables)

        # Some NN api_methods (3 input nodes, batch size=3).
        states = np.array([[-0.01, 0.02, -0.03], [0.04, -0.05, 0.06], [-0.07, 0.08, -0.09]])
        # Raw NN-output (3 hidden nodes). All weights=1.5, no biases.
        expected_nn_output = np.matmul(states, policy_params["policy/test-network/hidden-layer/dense/kernel"])
        expected_nn_output = relu(expected_nn_output, 0.1)
        test.test(("get_nn_output", states), expected_outputs=expected_nn_output)

        # Raw action layer output; Expected shape=(3,3): 3=batch, 2=action categories + 1 state value
        expected_action_layer_output = np.matmul(
            expected_nn_output, policy_params["policy/action-adapter/action-layer/dense/kernel"]
        )
        expected_action_layer_output = np.reshape(expected_action_layer_output, newshape=(3, 3))
        test.test(("get_action_layer_output", states), expected_outputs=expected_action_layer_output)

        # State-values: One for each item in the batch (simply take first out-node of action_layer).
        expected_state_value_output = np.squeeze(expected_action_layer_output[:, :1], axis=-1)
        # Advantage-values: One for each action-choice per item in the batch (simply take second and third out-node
        expected_advantage_values_output = expected_action_layer_output[:, 1:]
        # Q-values: One for each action-choice per item in the batch (calculate from state-values and advantage-values
        expected_q_values_output = np.reshape(expected_state_value_output, newshape=(3, 1)) + \
                                   expected_advantage_values_output - \
                                   np.mean(expected_advantage_values_output, axis=-1, keepdims=True)
        test.test(("get_dueling_output", states), expected_outputs=[
            expected_state_value_output,
            expected_advantage_values_output,
            expected_q_values_output
        ])

        # Parameter (probabilities). Softmaxed q_values.
        expected_probabilities_output = softmax(expected_q_values_output, axis=-1)
        test.test(("get_logits_parameters_log_probs", states, [0, 1]),
                  expected_outputs=[
                      expected_q_values_output,
                      expected_probabilities_output
                      #np.log(expected_probabilities_output),
                  ])

        print("Probs: {}".format(expected_probabilities_output))

        # Stochastic sample.
        expected_actions = np.array([0, 0, 1])
        test.test(("sample_stochastic", states), expected_outputs=expected_actions)

        # Deterministic sample.
        expected_actions = np.array([0, 1, 0])
        test.test(("sample_deterministic", states), expected_outputs=expected_actions)

        # Distribution's entropy.
        expected_h = np.array([0.6931462, 0.6925241, 0.6931312])
        test.test(("get_entropy", states), expected_outputs=expected_h)
