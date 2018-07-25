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

from rlgraph.components.neural_networks.action_adapter import ActionAdapter
from rlgraph.spaces import *
from rlgraph.tests import ComponentTest


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
        test = ComponentTest(component=action_adapter, input_spaces=dict(
            get_action_layer_output=last_nn_layer_space,
            get_logits_parameters_log_probs=last_nn_layer_space
        ), action_space=action_space)

        # Batch of 2 samples.
        inputs = np.array([
            [0.0, 0.1, 0.2, 0.3, 2.1, 0.4, 0.5, 0.6, 2.2, 0.7, 0.8, 0.9, 2.3, 1.0, 1.1, 1.2],
            [-1.0, -1.1, -1.2, -1.3, -3.1, -1.4, -1.5, -1.6, -3.2, -1.7, -1.8, -1.9, 3.3, 2.0, 2.1, 2.2],
        ])
        expected_action_layer_output = np.array([
            [14.4] * 12,
            [0.0] * 12,
        ], dtype=np.float32)
        test.test(("get_action_layer_output", inputs), expected_outputs=expected_action_layer_output)

        expected_logits = np.reshape(expected_action_layer_output, newshape=(2, 3, 2, 2))
        expected_parameters = np.array([
            [[[0.5] * 2] * 2] * 3,
            [[[0.5] * 2] * 2] * 3,
            ], dtype=np.float32)
        expected_log_probs = np.log(expected_parameters)
        test.test(("get_logits_parameters_log_probs", inputs),
                  expected_outputs=[expected_logits, expected_parameters, expected_log_probs])

    def test_action_adapter_with_dueling_layer(self):
        # Last NN layer.
        last_nn_layer_space = FloatBox(shape=(7,), add_batch_rank=True)
        # Action Space.
        action_space = IntBox(4, shape=(2,))

        action_adapter = ActionAdapter(action_space=action_space, weights_spec=2.0, biases_spec=0.5,
                                       activation="linear", add_dueling_layer=True)
        test = ComponentTest(component=action_adapter, input_spaces=dict(
            get_action_layer_output=last_nn_layer_space,
            get_dueling_output=last_nn_layer_space,
            get_logits_parameters_log_probs=last_nn_layer_space
        ), action_space=action_space)

        # Batch of 2 samples.
        inputs = np.array([[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]])
        # 1: state value
        # 8: advantage values for the flattened action space.
        # 2.0 (weights) * SUM(api_methods) + 0.5 (bias) = 4.7 for first sample, 18.7 for second sample in batch.
        expected_action_layer_output = np.array([[4.7] * (1+8), [18.7] * (1+8)], dtype=np.float32)
        test.test(("get_action_layer_output", inputs), expected_outputs=expected_action_layer_output)

        expected_state_advantage_q_values = [
            np.array([4.7, 18.7], dtype=np.float32),
            np.array([[[4.7] * 4] * 2, [[18.7] * 4] * 2], dtype=np.float32),
            np.array([[[4.7] * 4] * 2, [[18.7] * 4] * 2], dtype=np.float32)
        ]
        test.test(("get_dueling_output", inputs), expected_outputs=expected_state_advantage_q_values)

