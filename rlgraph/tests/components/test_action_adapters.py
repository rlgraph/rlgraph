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

import unittest
import numpy as np

from rlgraph.components.action_adapters.action_adapter import ActionAdapter
from rlgraph.components.action_adapters.baseline_action_adapter import BaselineActionAdapter
from rlgraph.components.action_adapters.dueling_action_adapter import DuelingActionAdapter
from rlgraph.spaces import *
from rlgraph.tests import ComponentTest
from rlgraph.utils.numpy import softmax, relu


class TestActionAdapters(unittest.TestCase):
    """
    Tests for the different ActionAdapter setups.
    """
    def test_simple_action_adapter(self):
        # Last NN layer.
        last_nn_layer_space = FloatBox(shape=(16,), add_batch_rank=True)
        # Action Space.
        action_space = IntBox(2, shape=(3, 2))

        action_adapter = ActionAdapter(action_space=action_space, weights_spec=1.0, biases_spec=False,
                                       activation="relu")
        test = ComponentTest(
            component=action_adapter, input_spaces=dict(nn_output=last_nn_layer_space), action_space=action_space
        )
        action_adapter_params = test.read_variable_values(action_adapter.variables)

        # Batch of 2 samples.
        inputs = last_nn_layer_space.sample(2)

        expected_action_layer_output = np.matmul(
            inputs, action_adapter_params["action-adapter/action-layer/dense/kernel"]
        )
        test.test(("get_action_layer_output", inputs), expected_outputs=dict(output=expected_action_layer_output))

        expected_logits = np.reshape(expected_action_layer_output, newshape=(2, 3, 2, 2))
        expected_probabilities = softmax(expected_logits)
        expected_log_probs = np.log(expected_probabilities)
        test.test(("get_logits_probabilities_log_probs", inputs), expected_outputs=dict(
            logits=expected_logits, probabilities=expected_probabilities, log_probs=expected_log_probs
        ))

    def test_action_adapter_with_complex_lstm_output(self):
        # Last NN layer (LSTM with time rank).
        last_nn_layer_space = FloatBox(shape=(4,), add_batch_rank=True, add_time_rank=True, time_major=True)
        # Action Space.
        action_space = IntBox(2, shape=(3, 2))

        action_adapter = ActionAdapter(action_space=action_space, biases_spec=False)
        test = ComponentTest(
            component=action_adapter, input_spaces=dict(nn_output=last_nn_layer_space), action_space=action_space
        )
        action_adapter_params = test.read_variable_values(action_adapter.variables)

        # Batch of 2 samples, 3 timesteps.
        inputs = last_nn_layer_space.sample(size=(3, 2))
        # Fold time rank before the action layer pass through.
        inputs_reshaped = np.reshape(inputs, newshape=(6, -1))
        # Action layer pass through and unfolding of time rank.
        expected_action_layer_output = np.matmul(
            inputs_reshaped, action_adapter_params["action-adapter/action-layer/dense/kernel"]
        ).reshape((3, 2, -1))
        test.test(("get_action_layer_output", inputs), expected_outputs=dict(output=expected_action_layer_output))

        # Logits (already well reshaped (same as action space)).
        expected_logits = np.reshape(expected_action_layer_output, newshape=(3, 2, 3, 2, 2))
        # Softmax (probs).
        expected_probabilities = softmax(expected_logits)
        # Log probs.
        expected_log_probs = np.log(expected_probabilities)
        test.test(("get_logits_probabilities_log_probs", inputs), expected_outputs=dict(
            logits=expected_logits, probabilities=expected_probabilities, log_probs=expected_log_probs
        ), decimals=5)

    def test_dueling_action_adapter(self):
        # Last NN layer.
        last_nn_layer_space = FloatBox(shape=(7,), add_batch_rank=True)
        # Action Space.
        action_space = IntBox(4, shape=(2,))

        action_adapter = DuelingActionAdapter(
            action_space=action_space, units_state_value_stream=5, units_advantage_stream=4,
            weights_spec_state_value_stream=1.0, weights_spec_advantage_stream=0.5,
            activation_advantage_stream="linear",
            scope="aa"
        )
        test = ComponentTest(
            component=action_adapter, input_spaces=dict(nn_output=last_nn_layer_space), action_space=action_space
        )

        # Batch of 2 samples.
        batch_size = 2
        inputs = last_nn_layer_space.sample(size=batch_size)

        dueling_action_adapter_vars = test.read_variable_values(action_adapter.variables)

        # Expected action layer output are the advantage nodes.
        expected_raw_advantages = np.matmul(np.matmul(
            inputs, dueling_action_adapter_vars["aa/dense-layer-advantage-stream/dense/kernel"]
        ), dueling_action_adapter_vars["aa/action-layer/dense/kernel"])
        expected_state_values = np.matmul(relu(np.matmul(
            inputs, dueling_action_adapter_vars["aa/dense-layer-state-value-stream/dense/kernel"]
        )), dueling_action_adapter_vars["aa/state-value-node/dense/kernel"])

        test.test(("get_action_layer_output", inputs), expected_outputs=dict(
            state_value_node=expected_state_values, output=expected_raw_advantages
        ), decimals=5)

        expected_advantages = np.reshape(expected_raw_advantages, newshape=(batch_size, 2, 4))

        # Expected q-values/logits, probabilities (softmaxed q) and log(p).
        expanded_state_values = np.expand_dims(expected_state_values, axis=1)
        expected_q_values = expanded_state_values + expected_advantages - \
            np.mean(expected_advantages, axis=-1, keepdims=True)
        expected_probs = softmax(expected_q_values)

        test.test(("get_logits_probabilities_log_probs", inputs), expected_outputs=dict(
            state_values=expected_state_values,
            logits=expected_q_values,
            probabilities=expected_probs,
            log_probs=np.log(expected_probs)
        ), decimals=3)

    """
    def test_dueling_layer(self):
        # Action Space is: IntBox(3, shape=(4,2)) ->
        # Flat input space to dueling layer is then 3x4x2 + 1: FloatBox(shape=(25,)).
        input_space = FloatBox(shape=(25,), add_batch_rank=True)
        action_space = IntBox(3, shape=(4, 2))

        dueling_layer = DuelingLayer()
        test = ComponentTest(component=dueling_layer, input_spaces=dict(inputs=input_space), action_space=action_space)

        # Batch of 1 sample.
        inputs = np.array(
            [[2.12345, 0.1, 0.2, 0.3, 2.1, 0.4, 0.5, 0.6, 2.2, 0.7, 0.8, 0.9, 2.3, 1.0, 1.1, 1.2, 2.4, 1.3, 1.4, 1.5,
              2.5, 1.6, 1.7, 1.8, 2.6
              ]]
        )
        ""
         Calculation: Very 1st node is the state-value, all others are the advantages per action.
        ""
        expected_state_value = np.array([2.12345])  # batch-size=1
        expected_advantage_values = np.reshape(inputs[:,1:], newshape=(1, 4, 2, 3))
        expected_q_values = np.array([[[[ expected_state_value[0] ]]]]) + expected_advantage_values - \
                            np.mean(expected_advantage_values, axis=-1, keepdims=True)
        test.test(("apply", inputs),
                  expected_outputs=[expected_state_value,
                                    expected_advantage_values,
                                    expected_q_values],
                  decimals=5)
    """

    def test_baseline_action_adapter(self):
        # Last NN layer.
        last_nn_layer_space = FloatBox(shape=(8,), add_batch_rank=True)
        # Action Space.
        action_space = IntBox(2, shape=(2, 2))

        action_adapter = BaselineActionAdapter(action_space=action_space, activation="linear")
        test = ComponentTest(
            component=action_adapter, input_spaces=dict(nn_output=last_nn_layer_space), action_space=action_space
        )

        # Read out weights and bias. Should have been initialized randomly.
        action_layer_vars = test.read_variable_values(action_adapter.action_layer.variables)

        # Batch of 3 samples (9 nodes each: 1 for state-value node AND 2x2x2 for actions).
        nn_output = last_nn_layer_space.sample(3)

        # Raw action layer output.
        expected_action_layer_output = np.matmul(
            nn_output, action_layer_vars["baseline-action-adapter/action-layer/dense/kernel"]
        ) + action_layer_vars["baseline-action-adapter/action-layer/dense/bias"]

        test.test(("get_action_layer_output", nn_output), expected_outputs=dict(output=expected_action_layer_output),
                  decimals=5)

        expected_state_values = expected_action_layer_output[:, 0:1]
        expected_action_logits = np.reshape(expected_action_layer_output[:, 1:], newshape=(-1, 2, 2, 2))

        test.test(("get_state_values_and_logits", nn_output), expected_outputs=(
            expected_state_values, expected_action_logits
        ), decimals=5)

