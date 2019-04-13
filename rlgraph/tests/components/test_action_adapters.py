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

from rlgraph.components.action_adapters import *
from rlgraph.spaces import *
from rlgraph.tests import ComponentTest
from rlgraph.utils.numpy import softmax


class TestActionAdapters(unittest.TestCase):
    """
    Tests for the different ActionAdapter setups.
    """
    def test_simple_action_adapter(self):
        # Last NN layer.
        previous_nn_layer_space = FloatBox(shape=(16,), add_batch_rank=True)
        adapter_outputs_space = FloatBox(shape=(3, 2, 2), add_batch_rank=True)
        # Action Space.
        action_space = IntBox(2, shape=(3, 2))

        action_adapter = CategoricalDistributionAdapter(action_space=action_space, weights_spec=1.0, biases_spec=False,
                                                        activation="relu")
        test = ComponentTest(
            component=action_adapter, input_spaces=dict(
                inputs=previous_nn_layer_space,
                adapter_outputs=adapter_outputs_space,
            ), action_space=action_space
        )
        action_adapter_params = test.read_variable_values(action_adapter.variable_registry)

        # Batch of 2 samples.
        inputs = previous_nn_layer_space.sample(2)

        expected_action_layer_output = np.matmul(
            inputs, action_adapter_params["action-adapter/action-network/action-layer/dense/kernel"]
        )
        expected_logits = np.reshape(expected_action_layer_output, newshape=(2, 3, 2, 2))
        test.test(("call", inputs), expected_outputs=expected_logits, decimals=5)
        #test.test(("get_logits", inputs), expected_outputs=expected_logits, decimals=5)  # w/o the dict

        expected_parameters = softmax(expected_logits)
        expected_log_probs = np.log(expected_parameters)
        test.test(("get_parameters", inputs), expected_outputs=dict(
            adapter_outputs=expected_logits, parameters=expected_parameters, log_probs=expected_log_probs
        ), decimals=5)

    def test_simple_action_adapter_with_batch_apply(self):
        # Last NN layer.
        previous_nn_layer_space = FloatBox(shape=(16,), add_batch_rank=True, add_time_rank=True, time_major=True)
        adapter_outputs_space = FloatBox(shape=(3, 2, 2), add_batch_rank=True)
        # Action Space.
        action_space = IntBox(2, shape=(3, 2))

        action_adapter = CategoricalDistributionAdapter(
            action_space=action_space, weights_spec=1.0, biases_spec=False, fold_time_rank=True, unfold_time_rank=True,
            activation="relu"
        )
        test = ComponentTest(
            component=action_adapter, input_spaces=dict(
                inputs=previous_nn_layer_space,
                adapter_outputs=adapter_outputs_space
            ), action_space=action_space
        )
        action_adapter_params = test.read_variable_values(action_adapter.variable_registry)

        # Batch of (4, 5).
        inputs = previous_nn_layer_space.sample(size=(4, 5))
        inputs_folded = np.reshape(inputs, newshape=(20, -1))

        expected_action_layer_output = np.matmul(
            inputs_folded, action_adapter_params["action-adapter/action-network/action-layer/dense/kernel"]
        )
        expected_logits = np.reshape(expected_action_layer_output, newshape=(4, 5, 3, 2, 2))

        test.test(("call", inputs), expected_outputs=expected_logits, decimals=4)

        expected_parameters = softmax(expected_logits)
        expected_log_probs = np.log(expected_parameters)
        test.test(("get_parameters", inputs), expected_outputs=dict(
            adapter_outputs=expected_logits, parameters=expected_parameters, log_probs=expected_log_probs
        ), decimals=4)

    def test_action_adapter_with_complex_lstm_output(self):
        # Last NN layer (LSTM with time rank).
        previous_nn_layer_space = FloatBox(shape=(4,), add_batch_rank=True, add_time_rank=True, time_major=True)
        adapter_outputs_space = FloatBox(shape=(3, 2, 2), add_batch_rank=True)
        # Action Space.
        action_space = IntBox(2, shape=(3, 2))

        action_adapter = CategoricalDistributionAdapter(action_space=action_space, biases_spec=False)
        test = ComponentTest(
            component=action_adapter, input_spaces=dict(
                inputs=previous_nn_layer_space,
                adapter_outputs=adapter_outputs_space
            ), action_space=action_space
        )
        action_adapter_params = test.read_variable_values(action_adapter.variable_registry)

        # Batch of 2 samples, 3 timesteps.
        inputs = previous_nn_layer_space.sample(size=(3, 2))
        # Fold time rank before the action layer pass through.
        inputs_reshaped = np.reshape(inputs, newshape=(6, -1))
        # Action layer pass through and unfolding of time rank.
        expected_action_layer_output = np.matmul(
            inputs_reshaped, action_adapter_params["action-adapter/action-network/action-layer/dense/kernel"]
        ).reshape((3, 2, -1))
        # Logits (already well reshaped (same as action space)).
        expected_logits = np.reshape(expected_action_layer_output, newshape=(3, 2, 3, 2, 2))
        test.test(("call", inputs), expected_outputs=expected_logits)
        #test.test(("get_logits", inputs), expected_outputs=expected_logits)

        # Softmax (probs).
        expected_parameters = softmax(expected_logits)
        # Log probs.
        expected_log_probs = np.log(expected_parameters)
        test.test(("get_parameters", inputs), expected_outputs=dict(
            adapter_outputs=expected_logits, parameters=expected_parameters, log_probs=expected_log_probs
        ), decimals=5)

