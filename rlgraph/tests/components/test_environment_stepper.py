# Copyright 2018 The RLgraph authors, All Rights Reserved.
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
import time
import unittest

from rlgraph.environments.environment import Environment
from rlgraph.components.explorations.exploration import Exploration
from rlgraph.components.neural_networks.actor_component import ActorComponent
from rlgraph.components.common.environment_stepper import EnvironmentStepper
from rlgraph.spaces import FloatBox, IntBox, Tuple
from rlgraph.tests import ComponentTest
from rlgraph.tests.test_util import config_from_path
from rlgraph.utils.ops import DataOpTuple


class TestEnvironmentStepper(unittest.TestCase):
    """
    Tests for the EnvironmentStepper Component using a simple RandomEnv.
    """
    internal_states_space = Tuple(FloatBox(shape=(256,)), FloatBox(shape=(256,)), add_batch_rank=True)
    internal_states_space_test_lstm = Tuple(FloatBox(shape=(3,)), FloatBox(shape=(3,)), add_batch_rank=True)

    action_probs_space = FloatBox(shape=(4,), add_batch_rank=True)

    time_steps = 500

    def test_environment_stepper_on_random_env(self):
        np.random.seed(10)
        state_space = FloatBox(shape=(1,))
        action_space = IntBox(2)
        preprocessor_spec = None
        network_spec = config_from_path("configs/test_simple_nn.json")
        exploration_spec = None
        actor_component = ActorComponent(
            preprocessor_spec, dict(network_spec=network_spec, action_space=action_space), exploration_spec
        )
        environment_stepper = EnvironmentStepper(
            environment_spec=dict(
                type="random_env", state_space=state_space, action_space=action_space, deterministic=True
            ),
            actor_component_spec=actor_component,
            state_space=state_space,
            reward_space="float32",
            num_steps=3
        )

        test = ComponentTest(
            component=environment_stepper,
            action_space=action_space,
        )

        # Reset the stepper.
        test.test("reset")

        # Step 3 times through the Env and collect results.
        expected = (
            np.array([True, False, False, False]),  # t_
            np.array([[0.77132064], [0.74880385], [0.19806287], [0.08833981]]),  # s' (raw)
        )
        test.test("step", expected_outputs=expected)

        # Step again, check whether stitching of states/etc.. works.
        expected = (
            np.array([True, False, False, False]),  # t_
            np.array([[0.77132064], [0.00394827], [0.61252606], [0.91777414]]),  # s' (raw)
        )
        test.test("step", expected_outputs=expected)

        # Make sure we close the session (to shut down the Env on the server).
        test.terminate()

    def test_environment_stepper_on_random_env_with_returning_action_probs(self):
        state_space = FloatBox(shape=(2,))
        action_space = IntBox(4)
        preprocessor_spec = [dict(type="multiply", factor=3)]
        network_spec = config_from_path("configs/test_simple_nn.json")
        exploration_spec = None
        actor_component = ActorComponent(
            preprocessor_spec, dict(network_spec=network_spec, action_space=action_space), exploration_spec
        )
        environment_stepper = EnvironmentStepper(
            environment_spec=dict(
                type="random_env", state_space=state_space, action_space=action_space, deterministic=True
            ),
            actor_component_spec=actor_component,
            state_space=state_space,
            reward_space="float32",
            add_action_probs=True,
            action_probs_space=self.action_probs_space,
            num_steps=3
        )

        test = ComponentTest(
            component=environment_stepper,
            action_space=action_space,
        )

        np.random.seed(10)

        # Reset the stepper.
        test.test("reset")

        # Step 3 times through the Env and collect results.
        expected = (
            # t_
            np.array([True, False, False, False]),
            # s' (raw)
            np.array([[0.7713206, 0.0207519], [0.498507, 0.2247967], [0.1691108, 0.0883398], [0.0039483, 0.5121922]]),
            # action probs
            np.array([[0.0, 0.0, 0.0, 0.0],
                      [0.3181699, 0.0138463, 0.0614877, 0.606496],
                      [0.2051629, 0.0250753, 0.0531823, 0.7165795],
                      [0.2692995, 0.1314247, 0.1675677, 0.4317082]])
        )
        print(test.test("step", expected_outputs=None))

        # Step again, check whether stitching of states/etc.. works.
        expected = (
            np.array([True, False, False, False]),
            np.array([[0.77132064, 0.02075195], [0.7217553, 0.29187608], [0.54254436, 0.14217004],
                      [0.44183317, 0.434014]]),  # s' (raw)
            np.array([[0.0, 0.0, 0.0, 0.0],
                      [0.31817, 0.01385, 0.06149, 0.6065],
                      [0.1526, 0.00736, 0.02262, 0.81741],
                      [0.25166, 0.02651, 0.06657, 0.65526]])
        )
        test.test("step", expected_outputs=expected, decimals=5)

        # Make sure we close the session (to shut down the Env on the server).
        test.terminate()

    def test_environment_stepper_on_random_env_with_action_probs_lstm(self):
        np.random.seed(10)
        state_space = FloatBox(shape=(2,))
        action_space = IntBox(4)
        internal_states_space = Tuple(FloatBox(shape=(3,)), FloatBox(shape=(3,)))
        preprocessor_spec = [dict(type="multiply", factor=3)]
        network_spec = config_from_path("configs/test_lstm_nn.json")
        exploration_spec = None
        actor_component = ActorComponent(
            preprocessor_spec, dict(network_spec=network_spec, action_space=action_space), exploration_spec
        )
        environment_stepper = EnvironmentStepper(
            environment_spec=dict(
                type="random_env", state_space=state_space, action_space=action_space, deterministic=True
            ),
            actor_component_spec=actor_component,
            state_space=state_space,
            reward_space="float32",
            internal_states_space=internal_states_space,
            add_action_probs=True,
            action_probs_space=self.action_probs_space,
            num_steps=3,
        )

        test = ComponentTest(
            component=environment_stepper,
            action_space=action_space,
        )

        # Reset the stepper.
        test.test("reset")

        # Step 3 times through the Env and collect results.
        expected = (
            np.array([True, False, False, False]),
            np.array([[0.77132064, 0.02075195], [0.49850702, 0.22479665], [0.16911083, 0.08833981],
                      [0.00394827, 0.51219225]]),  # s' (raw)
            np.array([[0.0, 0.0, 0.0, 0.0],
                      [0.29184222, 0.27833143, 0.20141664, 0.22840966],
                      [0.31360343, 0.28261214, 0.18345872, 0.22032563],
                      [0.30973074, 0.28645274, 0.18394111, 0.2198754]]),  # action probs
            # internal states
            (
                np.array([[0.0, 0.0, 0.0],
                          [0.17770568, -0.02882081, -0.44086117],
                          [0.30588162, 0.02668203, -0.7707858 ],
                          [0.25770405, 0.00710323, -0.81886315]]),
                np.array([[0.0, 0.0, 0.0],
                          [0.10143799, -0.01267258, -0.27671543],
                          [0.18374527, 0.01430159, -0.3720673],
                          [0.1395069, 0.00376055, -0.37974578]])
            )
        )
        print(test.test("step", expected_outputs=None))

        # Make sure we close the session (to shut down the Env on the server).
        test.terminate()

    def test_environment_stepper_on_pong(self):
        environment_spec = dict(type="openai_gym", gym_env="Pong-v0", frameskip=4, seed=10)
        dummy_env = Environment.from_spec(environment_spec)
        state_space = dummy_env.state_space
        action_space = dummy_env.action_space
        agent_config = config_from_path("configs/dqn_agent_for_pong.json")
        actor_component = ActorComponent(
            agent_config["preprocessing_spec"],
            dict(network_spec=agent_config["network_spec"],
                 action_adapter_spec=agent_config["action_adapter_spec"],
                 action_space=action_space),
            agent_config["exploration_spec"]
        )
        environment_stepper = EnvironmentStepper(
            environment_spec=environment_spec,
            actor_component_spec=actor_component,
            state_space=state_space,
            reward_space="float",
            num_steps=self.time_steps
        )

        test = ComponentTest(
            component=environment_stepper,
            action_space=action_space,
        )

        # Step 30 times through the Env and collect results.
        # 1st return value is the step-op (None), 2nd return value is the tuple of items (3 steps each), with each
        # step containing: Preprocessed state, actions, rewards, episode returns, terminals, (raw) next-states.
        # Reset the stepper.
        test.test("reset")
        time_start = time.monotonic()
        out = test.test("step")
        time_end = time.monotonic()
        print("Done running {} steps in env-stepper env in {}sec.".format(
            environment_stepper.num_steps, time_end - time_start
        ))

        # Check types of outputs.
        self.assertTrue(isinstance(out, DataOpTuple))  # the step results as a tuple (see below)

        # Check types of single data.
        #self.assertTrue(out[0].dtype == np.float32)  # preprocessed states
        #self.assertTrue(out[0].min() >= 0.0)  # make sure we have pixels / 255
        #self.assertTrue(out[0].max() <= 1.0)
        #self.assertTrue(out[1].dtype == np.int32)  # actions
        #self.assertTrue(out[2].dtype == np.float32)  # rewards
        #self.assertTrue(out[3].dtype == np.float32)  # episode return
        self.assertTrue(out[0].dtype == np.bool_)  # next-state is terminal?
        self.assertTrue(out[1].dtype == np.uint8)  # next state (raw, not preprocessed)
        self.assertTrue(out[1].min() >= 0)  # make sure we have pixels
        self.assertTrue(out[1].max() <= 255)

        # Check whether episode returns match single rewards (including resetting after each terminal signal).
        #episode_returns = 0.0
        #for i in range(environment_stepper.num_steps):
        #    episode_returns += out[2][i]
        #    self.assertAlmostEqual(episode_returns, out[1][3][i])
        #    # Terminal: Reset accumulated episode-return before next step.
        #    if out[1][4][i] is np.bool_(True):
        #        episode_returns = 0.0

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
                 action_adapter_spec=agent_config["action_adapter_spec"],
                 action_space=action_space),
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
            #preprocessed_s = out["preprocessed_state"]
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
                type="linear_decay", from_=1.0, to_=0.1, start_timestep=0, num_timesteps=100)
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
            #add_previous_action=True,
            #add_previous_reward=True,
            add_action_probs=True,
            action_probs_space=FloatBox(shape=(9,), add_batch_rank=True)
        )

        test = ComponentTest(
            component=environment_stepper,
            action_space=action_space,
        )
        # Reset the stepper.
        test.test("reset")

        # Step n times through the Env and collect results.
        # 1st return value is the step-op (None), 2nd return value is the tuple of items (3 steps each), with each
        # step containing: Preprocessed state, actions, rewards, episode returns, terminals, (raw) next-states.
        time_start = time.monotonic()
        steps = 10
        for _ in range(steps):
            out = test.test("step")
        time_total = time.monotonic() - time_start
        print("Done running {}x{} steps in Deepmind Lab env using IMPALA network in {}sec. ({} actions/sec)".format(
            steps, environment_stepper.num_steps, time_total, environment_stepper.num_steps * steps / time_total)
        )

        # Check types of outputs.
        self.assertTrue(isinstance(out, DataOpTuple))  # the step results as a tuple (see below)

        # Check types of single data.
        #self.assertTrue(out[0].dtype == np.float32)
        #self.assertTrue(out[0].min() >= 0.0)  # make sure we have pixels / 255
        #self.assertTrue(out[0].max() <= 1.0)
        #self.assertTrue(out[1].dtype == np.int32)  # actions
        #self.assertTrue(out[2].dtype == np.float32)  # rewards
        #self.assertTrue(out[0].dtype == np.float32)  # episode return
        self.assertTrue(out[0].dtype == np.bool_)  # next-state is terminal?
        self.assertTrue(out[1].dtype == np.uint8)  # next state (raw, not preprocessed)
        self.assertTrue(out[1].min() >= 0)  # make sure we have pixels
        self.assertTrue(out[1].max() <= 255)
        # action probs (test whether sum to one).
        #self.assertTrue(out[1][6].dtype == np.float32)
        #self.assertTrue(out[1][6].min() >= 0.0)
        #self.assertTrue(out[1][6].max() <= 1.0)
        #recursive_assert_almost_equal(out[1][6].sum(axis=-1, keepdims=False),
        #                              np.ones(shape=(environment_stepper.num_steps,)), decimals=4)
        # internal states (c- and h-state)
        self.assertTrue(out[3][0].dtype == np.float32)
        self.assertTrue(out[3][1].dtype == np.float32)
        self.assertTrue(out[3][0].shape == (environment_stepper.num_steps, 3))
        self.assertTrue(out[3][1].shape == (environment_stepper.num_steps, 3))

        # Check whether episode returns match single rewards (including terminal signals).
        #episode_returns = 0.0
        #for i in range(environment_stepper.num_steps):
        #    episode_returns += out[0][i]
        #    self.assertAlmostEqual(episode_returns, out[3][i])
        #    # Terminal: Reset for next step.
        #    if out[4][i] is np.bool_(True):
        #        episode_returns = 0.0

        test.terminate()
