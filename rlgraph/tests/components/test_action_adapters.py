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

from __future__ import absolute_import, division, print_function

import unittest

import numpy as np

from rlgraph.components.action_adapters import BernoulliDistributionAdapter, CategoricalDistributionAdapter, \
    NormalMixtureDistributionAdapter
from rlgraph.spaces import *
from rlgraph.tests import ComponentTest
from rlgraph.utils.numpy import softmax, sigmoid, relu, dense_layer


class TestActionAdapters(unittest.TestCase):
    """
    Tests for the different ActionAdapter setups.
    """
    def test_bernoulli_distribution_adapter(self):
        # Last NN layer.
        previous_nn_layer_space = FloatBox(shape=(16,), add_batch_rank=True)
        adapter_outputs_space = FloatBox(shape=(2,), add_batch_rank=True)
        # Action Space.
        action_space = BoolBox(shape=(2,))

        action_adapter = BernoulliDistributionAdapter(action_space=action_space, activation="relu")
        test = ComponentTest(
            component=action_adapter, input_spaces=dict(
                inputs=previous_nn_layer_space,
                adapter_outputs=adapter_outputs_space,
            ), action_space=action_space
        )
        action_adapter_params = test.read_variable_values(action_adapter.variable_registry)

        # Batch of n samples.
        inputs = previous_nn_layer_space.sample(32)

        expected_logits = relu(np.matmul(
            inputs, action_adapter_params["action-adapter/action-network/action-layer/dense/kernel"]
        ))
        test.test(("call", inputs), expected_outputs=expected_logits, decimals=5)

        expected_probs = sigmoid(expected_logits)
        expected_log_probs = np.log(expected_probs)
        test.test(("get_parameters", inputs), expected_outputs=dict(
            adapter_outputs=expected_logits, parameters=expected_probs, probabilities=expected_probs,
            log_probs=expected_log_probs
        ), decimals=5)

    def test_categorical_distribution_adapter(self):
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

        expected_probs = softmax(expected_logits)
        expected_log_probs = np.log(expected_probs)
        test.test(("get_parameters", inputs), expected_outputs=dict(
            adapter_outputs=expected_logits, parameters=expected_logits, probabilities=expected_probs,
            log_probs=expected_log_probs
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

        expected_probs = softmax(expected_logits)
        expected_log_probs = np.log(expected_probs)
        test.test(("get_parameters", inputs), expected_outputs=dict(
            adapter_outputs=expected_logits, parameters=expected_logits, probabilities=expected_probs,
            log_probs=expected_log_probs
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
        expected_probs = softmax(expected_logits)
        # Log probs.
        expected_log_probs = np.log(expected_probs)
        test.test(("get_parameters", inputs), expected_outputs=dict(
            adapter_outputs=expected_logits, parameters=expected_logits, probabilities=expected_probs,
            log_probs=expected_log_probs
        ), decimals=5)

    def test_normal_mixture_distribution_adapter(self):
        # Some output space (4-variate Gaussian).
        num_events = 4
        output_space = FloatBox(shape=(num_events,), add_batch_rank=True)
        # How many sub-Distributions (each one a multivariate Gaussian) do we want to categorically sample from?
        num_mixtures = 3

        # Some arbitrary last layer of the preceding network.
        previous_nn_layer_space = FloatBox(shape=(50,), add_batch_rank=True)
        input_spaces = dict(
            inputs=[previous_nn_layer_space],
            adapter_outputs=FloatBox(shape=(num_mixtures + num_mixtures * 2 * num_events,), add_batch_rank=True)
        )

        distribution_adapter = NormalMixtureDistributionAdapter(
            action_space=output_space,
            num_mixtures=num_mixtures
        )

        test = ComponentTest(component=distribution_adapter, input_spaces=input_spaces)

        # Batch of size=n.
        input_ = input_spaces["inputs"][0].sample(10)

        global_scope = "normal-mixture-adapter/"
        # Calculate output manually.
        var_dict = test.read_variable_values(distribution_adapter.variable_registry)

        adapter_network_out = dense_layer(
            input_, var_dict[global_scope+"action-network/action-layer/dense/kernel"],
            var_dict[global_scope+"action-network/action-layer/dense/bias"]
        )
        test.test(("call", input_), expected_outputs=adapter_network_out, decimals=5)

        expected_parameters = dict(
            categorical=adapter_network_out[:, :num_mixtures],  # raw logits
            parameters0=tuple([
                adapter_network_out[:, num_mixtures:num_mixtures + num_events],  # mean
                np.exp(adapter_network_out[:, num_mixtures + num_events * 3:num_mixtures + num_events * 4]),  # sd
            ]),
            parameters1=tuple([
                adapter_network_out[:, num_mixtures + num_events:num_mixtures + num_events * 2],  # mean
                np.exp(adapter_network_out[:, num_mixtures + num_events * 4:num_mixtures + num_events * 5]),  # sd
            ]),
            parameters2=tuple([
                adapter_network_out[:, num_mixtures + num_events * 2:num_mixtures + num_events * 3],  # mean
                np.exp(adapter_network_out[:, num_mixtures + num_events * 5:num_mixtures + num_events * 6]),  # sd
            ]),
        )
        # Probs: Only for Categorical (softmax the logits).
        expected_probabilities = dict(categorical=softmax(expected_parameters["categorical"]))
        expected_log_probabilities = dict(categorical=np.log(expected_probabilities["categorical"]))

        test.test(
            ("get_parameters_from_adapter_outputs", adapter_network_out),
            expected_outputs=dict(parameters=expected_parameters, probabilities=expected_probabilities,
                                  log_probs=expected_log_probabilities),
            decimals=5
        )

        test.terminate()
