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

from rlgraph.components.neural_networks.actor_component import ActorComponent
from rlgraph.components.explorations.exploration import Exploration
from rlgraph.components.neural_networks.policy import Policy
from rlgraph.components.neural_networks.preprocessor_stack import PreprocessorStack
from rlgraph.spaces import *
from rlgraph.tests import ComponentTest
from rlgraph.tests.test_util import config_from_path
from rlgraph.utils import softmax, relu


class TestActorComponents(unittest.TestCase):

    def test_simple_actor_component(self):
        # state_space (NN is a simple single fc-layer relu network (2 units), random biases, random weights).
        state_space = FloatBox(shape=(5,), add_batch_rank=True)
        # action_space.
        action_space = IntBox(10, add_batch_rank=True)

        preprocessor = PreprocessorStack.from_spec(
            [dict(type="convert_type", to_dtype="float"), dict(type="multiply", factor=2)]
        )
        policy = Policy(neural_network=config_from_path("configs/test_simple_nn.json"), action_space=action_space)
        exploration = Exploration()  # no exploration
        actor_component = ActorComponent(preprocessor, policy, exploration)
        test = ComponentTest(
            component=actor_component,
            input_spaces=dict(states=state_space),
            action_space=action_space
        )
        # Some state inputs (5 input nodes, batch size=2).
        states = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        # Get and check some actions.
        actor_component_params = test.read_variable_values(actor_component.variables)
        # Expected NN-output.
        expected_nn_output = np.matmul(
            states * 2, actor_component_params["actor-component/policy/test-network/hidden-layer/dense/kernel"]
        )
        # Raw action layer output.
        expected_action_layer_output = np.matmul(
            expected_nn_output,
            actor_component_params["actor-component/policy/action-adapter/action-layer/dense/kernel"]
        )
        # Final actions (max-likelihood/greedy pick).
        expected_actions = np.argmax(expected_action_layer_output, axis=-1)
        expected_preprocessed_state = states * 2
        test.test(
            ("get_preprocessed_state_and_action", states),
            expected_outputs=(expected_preprocessed_state, expected_actions)
        )

    def test_actor_component_with_lstm_network(self):
        # state space and internal state space
        state_space = FloatBox(shape=(2,), add_batch_rank=True, add_time_rank=True, time_major=False)
        internal_state_space = Tuple(FloatBox(shape=(3,)), FloatBox(shape=(3,)), add_batch_rank=True)
        # action_space.
        action_space = IntBox(2, add_batch_rank=True, add_time_rank=True)

        preprocessor = PreprocessorStack.from_spec(
            [dict(type="convert_type", to_dtype="float"), dict(type="divide", divisor=10)]
        )
        policy = Policy(neural_network=config_from_path("configs/test_lstm_nn.json"), action_space=action_space)
        exploration = Exploration()  # no exploration
        actor_component = ActorComponent(preprocessor, policy, exploration)
        test = ComponentTest(
            component=actor_component,
            input_spaces=dict(states=state_space, internal_states=internal_state_space),
            action_space=action_space
        )
        # Some state inputs (batch size=2, seq-len=3; batch-major).
        np.random.seed(10)
        states = state_space.sample(size=(2, 3))
        initial_internal_states = internal_state_space.zeros(size=2)  # only batch

        # TODO: we need a numpy LSTM implementation (lift from our LSTMLayer test case) to be able to calculate manually in these test cases here
        expected_actions = np.array([[1, 1, 1], [1, 1, 1]])
        expected_preprocessed_state = states / 10
        expected_final_internal_states = (
            np.array([[-0.03173002, -0.03439367, 0.01098747],
                      [-0.02962531, -0.0488404, 0.00396963]]),
            np.array([[-0.01599146, -0.01695652, 0.00551831],
                      [-0.01525987, -0.02397921,  0.00199732]]),
        )
        test.test(
            ("get_preprocessed_state_and_action", [states, initial_internal_states]),
            expected_outputs=(expected_preprocessed_state, expected_actions, expected_final_internal_states)
        )
