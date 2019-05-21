# Copyright 2018/2019 The RLgraph authors, All Rights Reserved.
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

import time
import unittest

import numpy as np

from rlgraph.components.common.environment_stepper import EnvironmentStepper
from rlgraph.components.explorations.exploration import Exploration
from rlgraph.components.neural_networks.actor_component import ActorComponent
from rlgraph.environments.environment import Environment
from rlgraph.spaces import FloatBox, IntBox, Tuple
from rlgraph.tests import ComponentTest
from rlgraph.tests.test_util import config_from_path, recursive_assert_almost_equal
from rlgraph.utils.numpy import dense_layer, softmax, lstm_layer
from rlgraph.utils.ops import DataOpTuple


class TestEnvironmentStepper(unittest.TestCase):
    """
    Tests for the EnvironmentStepper Component using a simple RandomEnv.
    """
    deterministic_env_state_space = FloatBox(shape=(1,))
    deterministic_env_action_space = IntBox(2)
    deterministic_action_probs_space = FloatBox(shape=(2,), add_batch_rank=True)

    grid_world_2x2_state_space = IntBox(4)
    grid_world_2x2_action_space = IntBox(4)
    grid_world_2x2_action_probs_space = FloatBox(shape=(4,), add_batch_rank=True)

    internal_states_space = Tuple(FloatBox(shape=(256,)), FloatBox(shape=(256,)), add_batch_rank=True)
    internal_states_space_test_lstm = Tuple(FloatBox(shape=(3,)), FloatBox(shape=(3,)), add_batch_rank=True)

    action_probs_space = FloatBox(shape=(4,), add_batch_rank=True)

    time_steps = 500

    def test_environment_stepper_on_deterministic_env(self):
        preprocessor_spec = None
        network_spec = config_from_path("configs/test_simple_nn.json")
        exploration_spec = None
        actor_component = ActorComponent(
            preprocessor_spec,
            dict(network_spec=network_spec, action_space=self.deterministic_env_action_space),
            exploration_spec
        )
        environment_stepper = EnvironmentStepper(
            environment_spec=dict(type="deterministic_env", steps_to_terminal=5),
            actor_component_spec=actor_component,
            state_space=self.deterministic_env_state_space,
            reward_space="float32",
            num_steps=3
        )

        test = ComponentTest(
            component=environment_stepper,
            action_space=self.deterministic_env_action_space,
        )

        # Step 3 times through the Env and collect results.
        expected = (
            np.array([False, False, False]),  # t_
            np.array([[0.0], [1.0], [2.0], [3.0]]),  # s' (raw)
        )
        test.test("step", expected_outputs=expected)

        # Step again, check whether stitching of states/etc.. works.
        expected = (
            np.array([False, True, False]),  # t_
            np.array([[3.0], [4.0], [0.0], [1.0]]),  # s' (raw)
        )
        test.test("step", expected_outputs=expected)

        # Make sure we close the session (to shut down the Env on the server).
        test.terminate()

    def test_environment_stepper_on_2x2_grid_world(self):
        preprocessor_spec = [dict(
            type="reshape", flatten=True, flatten_categories=self.grid_world_2x2_action_space.num_categories
        )]
        network_spec = config_from_path("configs/test_simple_nn.json")
        # Try to find a NN that outputs greedy actions down in start state and right in state=1 (to reach goal).
        network_spec["layers"][0]["weights_spec"] = [[0.5, -0.5], [-0.1, 0.1], [-0.2, 0.2], [-0.4, 0.2]]
        network_spec["layers"][0]["biases_spec"] = False
        exploration_spec = None
        actor_component = ActorComponent(
            preprocessor_spec,
            dict(network_spec=network_spec, action_adapter_spec=dict(
                weights_spec=[[0.1, -0.5, 0.5, 0.1], [0.4, 0.2, -0.2, 0.2]],
                biases_spec=False
            ), action_space=self.grid_world_2x2_action_space, deterministic=True),
            exploration_spec
        )
        environment_stepper = EnvironmentStepper(
            environment_spec=dict(type="grid_world", world="2x2"),
            actor_component_spec=actor_component,
            state_space=self.grid_world_2x2_state_space,
            reward_space="float32",
            add_action_probs=True,
            action_probs_space=self.grid_world_2x2_action_probs_space,
            num_steps=5
        )

        test = ComponentTest(
            component=environment_stepper,
            action_space=self.grid_world_2x2_action_space,
        )

        # Step 5 times through the Env and collect results.
        expected = (
            np.array([False, True, False, True, False]),  # t_
            np.array([0, 1, 0, 1, 0, 1]),  # s' (raw)
            np.array([[0.21869287, 0.17905058, 0.36056358, 0.24169299],
                      [0.2547221, 0.2651175, 0.23048209, 0.24967825],
                      [0.21869287, 0.17905058, 0.36056358, 0.24169299],
                      [0.2547221, 0.2651175, 0.23048209, 0.24967825],
                      [0.21869287, 0.17905058, 0.36056358, 0.24169299]], dtype=np.float32)
        )
        out = test.test("step", expected_outputs=expected, decimals=2)
        print(out)

        # Step again, check whether stitching of states/etc.. works.
        expected = (
            np.array([True, False, True, False, True]),  # t_
            np.array([1, 0, 1, 0, 1, 0]),  # s' (raw)
            np.array([[0.2547221, 0.2651175, 0.23048209, 0.24967825],
                      [0.21869287, 0.17905058, 0.36056358, 0.24169299],
                      [0.2547221, 0.2651175, 0.23048209, 0.24967825],
                      [0.21869287, 0.17905058, 0.36056358, 0.24169299],
                      [0.2547221, 0.2651175, 0.23048209, 0.24967825]], dtype=np.float32)
        )
        out = test.test("step", expected_outputs=expected)
        print(out)

        # Make sure we close the session (to shut down the Env on the server).
        test.terminate()

    def test_environment_stepper_on_2x2_grid_world_returning_actions_and_rewards(self):
        preprocessor_spec = [dict(
            type="reshape", flatten=True, flatten_categories=self.grid_world_2x2_action_space.num_categories
        )]
        network_spec = config_from_path("configs/test_simple_nn.json")
        # Try to find a NN that outputs greedy actions down in start state and right in state=1 (to reach goal).
        network_spec["layers"][0]["weights_spec"] = [[0.5, -0.5], [-0.1, 0.1], [-0.2, 0.2], [-0.4, 0.2]]
        network_spec["layers"][0]["biases_spec"] = False
        exploration_spec = None
        actor_component = ActorComponent(
            preprocessor_spec,
            dict(network_spec=network_spec, action_adapter_spec=dict(
                weights_spec=[[0.1, -0.5, 0.5, 0.1], [0.4, 0.2, -0.2, 0.2]],
                biases_spec=False
            ), action_space=self.grid_world_2x2_action_space, deterministic=True),
            exploration_spec
        )
        environment_stepper = EnvironmentStepper(
            environment_spec=dict(type="grid_world", world="2x2"),
            actor_component_spec=actor_component,
            state_space=self.grid_world_2x2_state_space,
            reward_space="float32",
            add_action=True,
            add_reward=True,
            num_steps=5
        )

        test = ComponentTest(
            component=environment_stepper,
            action_space=self.grid_world_2x2_action_space,
        )

        # Step 5 times through the Env and collect results.
        expected = (
            np.array([False, True, False, True, False]),  # t_
            np.array([0, 1, 0, 1, 0, 1]),  # s' (raw)
            np.array([2, 1, 2, 1, 2]),  # actions taken
            np.array([-1.0, 1.0, -1.0, 1.0, -1.0])  # rewards
        )
        out = test.test("step", expected_outputs=expected, decimals=2)
        print(out)

        # Make sure we close the session (to shut down the Env on the server).
        test.terminate()

    def test_environment_stepper_on_deterministic_env_with_returning_action_probs(self):
        preprocessor_spec = [dict(type="divide", divisor=2)]
        network_spec = config_from_path("configs/test_simple_nn.json")
        exploration_spec = None
        actor_component = ActorComponent(
            preprocessor_spec,
            dict(network_spec=network_spec, action_space=self.deterministic_env_action_space),
            exploration_spec
        )
        environment_stepper = EnvironmentStepper(
            environment_spec=dict(type="deterministic_env", steps_to_terminal=6),
            actor_component_spec=actor_component,
            state_space=self.deterministic_env_state_space,
            reward_space="float32",
            add_action_probs=True,
            action_probs_space=self.deterministic_action_probs_space,
            num_steps=3
        )

        test = ComponentTest(
            component=environment_stepper,
            action_space=self.deterministic_env_action_space,
        )

        weights = test.read_variable_values(environment_stepper.actor_component.policy.variable_registry)
        policy_scope = "environment-stepper/actor-component/policy/"
        weights_hid = weights[policy_scope+"test-network/hidden-layer/dense/kernel"]
        biases_hid = weights[policy_scope+"test-network/hidden-layer/dense/bias"]
        weights_action = weights[policy_scope+"action-adapter-0/action-network/action-layer/dense/kernel"]
        biases_action = weights[policy_scope+"action-adapter-0/action-network/action-layer/dense/bias"]

        # Step 3 times through the Env and collect results.
        expected = (
            # t_
            np.array([False, False, False]),
            # s' (raw)
            np.array([[0.0], [1.0], [2.0], [3.0]]),
            # action probs
            np.array([
                softmax(dense_layer(dense_layer(np.array([0.0]), weights_hid, biases_hid), weights_action, biases_action)),
                softmax(dense_layer(dense_layer(np.array([0.5]), weights_hid, biases_hid), weights_action, biases_action)),
                softmax(dense_layer(dense_layer(np.array([1.0]), weights_hid, biases_hid), weights_action, biases_action))
            ])
        )
        test.test("step", expected_outputs=expected, decimals=3)

        # Step again, check whether stitching of states/etc.. works.
        expected = (
            np.array([False, False, True]),
            np.array([[3.0], [4.0], [5.0], [0.0]]),  # s' (raw)
            np.array([
                softmax(dense_layer(dense_layer(np.array([1.5]), weights_hid, biases_hid), weights_action, biases_action)),
                softmax(dense_layer(dense_layer(np.array([2.0]), weights_hid, biases_hid), weights_action, biases_action)),
                softmax(dense_layer(dense_layer(np.array([2.5]), weights_hid, biases_hid), weights_action, biases_action))
            ])
        )
        test.test("step", expected_outputs=expected, decimals=3)

        # Make sure we close the session (to shut down the Env on the server).
        test.terminate()

    def test_environment_stepper_on_deterministic_env_with_action_probs_lstm(self):
        internal_states_space = Tuple(FloatBox(shape=(3,)), FloatBox(shape=(3,)))
        preprocessor_spec = [dict(type="multiply", factor=0.1)]
        network_spec = config_from_path("configs/test_lstm_nn.json")
        exploration_spec = None
        actor_component = ActorComponent(
            preprocessor_spec,
            dict(network_spec=network_spec, action_space=self.deterministic_env_action_space),
            exploration_spec
        )
        environment_stepper = EnvironmentStepper(
            environment_spec=dict(type="deterministic_env", steps_to_terminal=3),
            actor_component_spec=actor_component,
            state_space=self.deterministic_env_state_space,
            reward_space="float32",
            internal_states_space=internal_states_space,
            add_action_probs=True,
            action_probs_space=self.deterministic_action_probs_space,
            num_steps=4,
        )

        test = ComponentTest(
            component=environment_stepper,
            action_space=self.deterministic_env_action_space,
        )

        weights = test.read_variable_values(environment_stepper.actor_component.policy.variable_registry)
        policy_scope = "environment-stepper/actor-component/policy/"
        weights_lstm = weights[policy_scope+"test-lstm-network/lstm-layer/lstm-cell/kernel"]
        biases_lstm = weights[policy_scope+"test-lstm-network/lstm-layer/lstm-cell/bias"]
        weights_action = weights[policy_scope+"action-adapter-0/action-network/action-layer/dense/kernel"]
        biases_action = weights[policy_scope+"action-adapter-0/action-network/action-layer/dense/bias"]

        # Step 3 times through the Env and collect results.
        lstm_1 = lstm_layer(np.array([[[0.0]]]), weights_lstm, biases_lstm)
        lstm_2 = lstm_layer(np.array([[[0.1]]]), weights_lstm, biases_lstm, lstm_1[1])
        lstm_3 = lstm_layer(np.array([[[0.2]]]), weights_lstm, biases_lstm, lstm_2[1])
        lstm_4 = lstm_layer(np.array([[[0.0]]]), weights_lstm, biases_lstm, lstm_3[1])
        expected = (
            np.array([False, False, True, False]),
            np.array([[0.0], [1.0], [2.0], [0.0], [1.0]]),  # s' (raw)
            np.array([
                softmax(dense_layer(np.squeeze(lstm_1[0]), weights_action, biases_action)),
                softmax(dense_layer(np.squeeze(lstm_2[0]), weights_action, biases_action)),
                softmax(dense_layer(np.squeeze(lstm_3[0]), weights_action, biases_action)),
                softmax(dense_layer(np.squeeze(lstm_4[0]), weights_action, biases_action)),
            ]),  # action probs
            # internal states
            (
                np.squeeze(np.array([[[0.0, 0.0, 0.0]], lstm_1[1][0], lstm_2[1][0], lstm_3[1][0], lstm_4[1][0]])),
                np.squeeze(np.array([[[0.0, 0.0, 0.0]], lstm_1[1][1], lstm_2[1][1], lstm_3[1][1], lstm_4[1][1]]))
            )
        )
        test.test("step", expected_outputs=expected)

        # Make sure we close the session (to shut down the Env on the server).
        test.terminate()

    def test_environment_stepper_on_pong(self):
        environment_spec = dict(type="openai-gym", gym_env="Pong-v0", frameskip=4, seed=10)
        dummy_env = Environment.from_spec(environment_spec)
        state_space = dummy_env.state_space
        action_space = dummy_env.action_space
        agent_config = config_from_path("configs/dqn_agent_for_pong.json")
        actor_component = ActorComponent(
            agent_config["preprocessing_spec"],
            dict(network_spec=agent_config["network_spec"],
                 action_space=action_space,
                 **agent_config["policy_spec"]),
            agent_config["exploration_spec"]
        )
        environment_stepper = EnvironmentStepper(
            environment_spec=environment_spec,
            actor_component_spec=actor_component,
            state_space=state_space,
            reward_space="float",
            add_reward=True,
            num_steps=self.time_steps
        )

        test = ComponentTest(
            component=environment_stepper,
            action_space=action_space,
        )

        # Step 30 times through the Env and collect results.
        # 1st return value is the step-op (None), 2nd return value is the tuple of items (3 steps each), with each
        # step containing: Preprocessed state, actions, rewards, episode returns, terminals, (raw) next-states.
        time_start = time.monotonic()
        out = test.test("step")
        time_end = time.monotonic()
        print("Done running {} steps in env-stepper env in {}sec.".format(
            environment_stepper.num_steps, time_end - time_start
        ))

        # Check types of outputs.
        self.assertTrue(isinstance(out, DataOpTuple))  # the step results as a tuple (see below)

        # Check types of single data.
        self.assertTrue(out[0].dtype == np.bool_)  # next-state is terminal?
        self.assertTrue(out[1].dtype == np.uint8)  # next state (raw, not preprocessed)
        self.assertTrue(out[1].min() >= 0)  # make sure we have pixels
        self.assertTrue(out[1].max() <= 255)
        self.assertTrue(out[2].dtype == np.float32)  # rewards
        self.assertTrue(out[2].min() >= -1.0)  # -1.0 to 1.0
        self.assertTrue(out[2].max() <= 1.0)

        # Make sure we close the session (to shut down the Env on the server).
        test.terminate()

    def test_compare_with_non_env_stepper(self):
        environment_spec = dict(type="openai_gym", gym_env="Pong-v0", frameskip=4, seed=10)
        dummy_env = Environment.from_spec(environment_spec)
        state_space = dummy_env.state_space.with_batch_rank()
        action_space = dummy_env.action_space
        agent_config = config_from_path("configs/dqn_agent_for_pong.json")
        actor_component = ActorComponent(
            agent_config["preprocessing_spec"],
            dict(network_spec=agent_config["network_spec"],
                 action_space=action_space,
                 **agent_config["policy_spec"]),
            agent_config["exploration_spec"]
        )
        test = ComponentTest(
            component=actor_component,
            input_spaces=dict(states=state_space),
            action_space=action_space,
        )
        s = dummy_env.reset()
        time_start = time.monotonic()
        for i in range(self.time_steps):
            out = test.test(("get_preprocessed_state_and_action", np.array([s])))
            a = out["action"]
            # Act in env.
            s, r, t, _ = dummy_env.step(a[0])  # remove batch
            if t is True:
                s = dummy_env.reset()
        time_end = time.monotonic()
        print("Done running {} steps in bare-metal env in {}sec.".format(self.time_steps, time_end - time_start))
        test.terminate()

    def test_environment_stepper_on_deepmind_lab(self):
        try:
            from rlgraph.environments.deepmind_lab import DeepmindLabEnv
        except ImportError:
            print("DeepmindLab not installed: Skipping this test case.")
            return

        env_spec = dict(
            type="deepmind_lab", level_id="seekavoid_arena_01", observations=["RGB_INTERLEAVED"], frameskip=4
        )
        dummy_env = Environment.from_spec(env_spec)
        state_space = dummy_env.state_space
        action_space = dummy_env.action_space
        actor_component = ActorComponent(
            # Preprocessor spec (only divide and flatten the image).
            [
                {
                    "type": "divide",
                    "divisor": 255
                },
                {
                    "type": "reshape",
                    "flatten": True
                }
            ],
            # Policy spec.
            dict(network_spec="../configs/test_lstm_nn.json", action_space=action_space),
            # Exploration spec.
            Exploration(epsilon_spec=dict(decay_spec=dict(
                type="linear_decay", from_=1.0, to_=0.1)
            ))
        )
        environment_stepper = EnvironmentStepper(
            environment_spec=env_spec,
            actor_component_spec=actor_component,
            state_space=state_space,
            reward_space="float32",
            internal_states_space=self.internal_states_space_test_lstm,
            num_steps=1000,
            # Add both prev-action and -reward into the state sent through the network.
            #add_previous_action_to_state=True,
            #add_previous_reward_to_state=True,
            add_action_probs=True,
            action_probs_space=FloatBox(shape=(9,), add_batch_rank=True)
        )

        test = ComponentTest(
            component=environment_stepper,
            action_space=action_space,
        )
        # Step n times through the Env and collect results.
        # 1st return value is the step-op (None), 2nd return value is the tuple of items (3 steps each), with each
        # step containing: Preprocessed state, actions, rewards, episode returns, terminals, (raw) next-states.
        time_start = time.monotonic()
        steps = 10
        out = None
        for _ in range(steps):
            out = test.test("step")
        time_total = time.monotonic() - time_start
        print("Done running {}x{} steps in Deepmind Lab env using IMPALA network in {}sec. ({} actions/sec)".format(
            steps, environment_stepper.num_steps, time_total, environment_stepper.num_steps * steps / time_total)
        )

        # Check types of outputs.
        self.assertTrue(isinstance(out, DataOpTuple))  # the step results as a tuple (see below)

        # Check types of single data.
        self.assertTrue(out[0].dtype == np.bool_)  # next-state is terminal?
        self.assertTrue(out[1].dtype == np.uint8)  # next state (raw, not preprocessed)
        self.assertTrue(out[1].min() >= 0)  # make sure we have pixels
        self.assertTrue(out[1].max() <= 255)
        # action probs (test whether sum to one).
        self.assertTrue(out[2].dtype == np.float32)
        self.assertTrue(out[2].min() >= 0.0)
        self.assertTrue(out[2].max() <= 1.0)
        recursive_assert_almost_equal(
            out[2].sum(axis=-1, keepdims=False), np.ones(shape=(environment_stepper.num_steps,)), decimals=4
        )
        # internal states (c- and h-state)
        self.assertTrue(out[3][0].dtype == np.float32)
        self.assertTrue(out[3][1].dtype == np.float32)
        self.assertTrue(out[3][0].shape == (environment_stepper.num_steps, 3))
        self.assertTrue(out[3][1].shape == (environment_stepper.num_steps, 3))

        test.terminate()
