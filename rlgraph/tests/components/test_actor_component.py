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
from rlgraph.tests.test_util import config_from_path, recursive_assert_almost_equal
from rlgraph.tests.dummy_components import DummyNNWithDictInput
from rlgraph.utils.numpy import softmax


class TestActorComponents(unittest.TestCase):

    def test_simple_actor_component(self):
        # state_space (NN is a simple single fc-layer relu network (2 units), random biases, random weights).
        state_space = FloatBox(shape=(5,), add_batch_rank=True)
        # action_space.
        action_space = IntBox(10)

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
        # Get and check some actions.
        actor_component_params = test.read_variable_values(actor_component.variables)

        # Some state inputs (5 input nodes, batch size=2).
        states = state_space.sample(2)
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

        # Get actions and action-probs by calling a different API-method.
        states = state_space.sample(5)
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
        # No reshape necessary (simple action space), softmax to get probs.
        expected_action_probs = softmax(expected_action_layer_output)
        # Final actions (max-likelihood/greedy pick).
        expected_actions = np.argmax(expected_action_layer_output, axis=-1)
        expected_preprocessed_state = states * 2
        test.test(
            ("get_preprocessed_state_action_and_action_probs", states),
            expected_outputs=(expected_preprocessed_state, expected_actions, expected_action_probs)
        )

    def test_actor_component_with_lstm_network(self):
        # state space and internal state space
        state_space = FloatBox(shape=(2,), add_batch_rank=True, add_time_rank=True, time_major=False)
        internal_states_space = Tuple(FloatBox(shape=(3,)), FloatBox(shape=(3,)), add_batch_rank=True)
        time_step_space = IntBox()
        # action_space.
        action_space = IntBox(2, add_batch_rank=True, add_time_rank=True)

        preprocessor = PreprocessorStack.from_spec(
            [dict(type="convert_type", to_dtype="float"), dict(type="divide", divisor=10)]
        )
        policy = Policy(neural_network=config_from_path("configs/test_lstm_nn.json"), action_space=action_space)
        exploration = Exploration(epsilon_spec=dict(decay_spec=dict(
            type="linear_decay", from_=1.0, to_=0.1, start_timestep=0, num_timesteps=100)
        ))
        actor_component = ActorComponent(preprocessor, policy, exploration)
        test = ComponentTest(
            component=actor_component,
            input_spaces=dict(
                states=state_space,
                internal_states=internal_states_space,
                time_step=time_step_space
            ),
            action_space=action_space
        )
        # Some state inputs (batch size=2, seq-len=1000; batch-major).
        np.random.seed(10)
        states = state_space.sample(size=(1000, 2))
        initial_internal_states = internal_states_space.zeros(size=2)  # only batch
        time_steps = time_step_space.sample(1000)

        # Run n times a single time-step to simulate acting and env interaction with an LSTM.
        preprocessed_states = np.ndarray(shape=(1000, 2, 2), dtype=np.float)
        actions = np.ndarray(shape=(1000, 2, 1), dtype=np.int)
        for i, time_step in enumerate(time_steps):
            ret = test.test((
                "get_preprocessed_state_and_action",
                # expand time dim at 1st slot as we are time-major == False
                [np.expand_dims(states[i], 1), initial_internal_states, time_step]
            ))
            preprocessed_states[i] = ret[0][:, 0, :]  # take out time-rank again ()
            actions[i] = ret[1]
            # Check c/h-state shape.
            self.assertEqual(ret[2][0].shape, (2, 3))  # batch-size=2, LSTM units=3
            self.assertEqual(ret[2][1].shape, (2, 3))

        # Check all preprocessed states (easy: just divided by 10).
        expected_preprocessed_state = states / 10
        recursive_assert_almost_equal(preprocessed_states, expected_preprocessed_state)

        # Check the exploration functionality over the actions.
        # Not checking mean as we are mostly in the non-exploratory region, that's why the stddev should be small.
        stddev_actions = actions.std()
        self.assertGreater(stddev_actions, 0.4)
        self.assertLess(stddev_actions, 0.6)

    def test_actor_component_with_dict_preprocessor(self):
        # state_space (a complex Dict Space, that will be partially preprocessed).
        state_space = Dict(
            a=FloatBox(shape=(2,)),
            b=FloatBox(shape=(5,)),
            add_batch_rank=True
        )
        # action_space.
        action_space = IntBox(2, add_batch_rank=True)

        preprocessor_spec = dict(
            type="dict-preprocessor-stack",
            preprocessors=dict(
                a=[dict(type="convert_type", to_dtype="float"), dict(type="multiply", factor=0.5)]
            )
        )
        # Simple custom NN with dict input (splits into 2 streams (simple dense layers) and concats at the end).
        policy = Policy(neural_network=DummyNNWithDictInput(num_units_a=2, num_units_b=3, scope="dummy-nn"),
                        action_space=action_space)
        exploration = None  # no exploration

        actor_component = ActorComponent(preprocessor_spec, policy, exploration)

        test = ComponentTest(
            component=actor_component,
            input_spaces=dict(states=state_space),
            action_space=action_space
        )

        # Some state inputs (batch size=4).
        states = state_space.sample(size=4)
        # Get and check some actions.
        actor_component_params = test.read_variable_values(actor_component.variables)
        # Expected NN-output.
        expected_nn_output_stream_a = np.matmul(
            states["a"] * 0.5, actor_component_params["actor-component/policy/dummy-nn/dense-a/dense/kernel"]
        )
        expected_nn_output_stream_b = np.matmul(
            states["b"], actor_component_params["actor-component/policy/dummy-nn/dense-b/dense/kernel"]
        )
        expected_nn_output = np.concatenate((expected_nn_output_stream_a, expected_nn_output_stream_b), axis=-1)

        # Raw action layer output.
        expected_action_layer_output = np.matmul(
            expected_nn_output,
            actor_component_params["actor-component/policy/action-adapter/action-layer/dense/kernel"]
        )
        # Final actions (max-likelihood/greedy pick).
        expected_actions = np.argmax(expected_action_layer_output, axis=-1)
        expected_preprocessed_state = dict(a=states["a"] * 0.5, b=states["b"])
        test.test(
            ("get_preprocessed_state_and_action", states),
            expected_outputs=(expected_preprocessed_state, expected_actions)
        )
