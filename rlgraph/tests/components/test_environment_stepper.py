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
    def test_environment_stepper_on_random_env(self):
        state_space = FloatBox(shape=(1,))
        action_space = IntBox(2)
        preprocessor_spec = None
        neural_network_spec = config_from_path("configs/test_simple_nn.json")
        exploration_spec = None
        actor_component = ActorComponent(
            preprocessor_spec, dict(neural_network=neural_network_spec, action_space=action_space), exploration_spec
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
        # 1st return value is the step-op (None), 2nd return value is the tuple of items (3 steps each), with each
        # step containing: Preprocessed state, actions, rewards, episode returns, terminals, (raw) next-states.
        expected_r = np.array([0.49850702, 0.7605307, 0.68535984])
        expected = (None, (
            np.array([[0.77132064], [0.74880385], [0.19806287]]),  # p(s)
            np.array([0, 0, 0]),  # a
            expected_r,  # r
            np.array([expected_r[:1].sum(), expected_r[:2].sum(), expected_r[:3].sum()]),  # episode's accumulated returns
            np.array([False, False, False]),
            np.array([[0.74880385], [0.19806287], [0.08833981]]),  # s' (raw)
        ))
        test.test("step", expected_outputs=expected)

        # Step again, check whether stitching of states/etc.. works.
        expected_r2 = np.array([0.51219225, 0.7217553, 0.71457577])
        expected = (None, (
            np.array([[0.08833981], [0.00394827], [0.61252606]]),  # p(s)
            np.array([0, 0, 0]),  # a
            expected_r2,  # r
            np.array([expected_r.sum() + expected_r2[0], expected_r.sum() + expected_r2[:2].sum(), expected_r.sum() + expected_r2[:3].sum()]),
            np.array([False, False, False]),
            np.array([[0.00394827], [0.61252606], [0.91777414]]),  # s' (raw)
        ))
        test.test("step", expected_outputs=expected)

        # Make sure we close the session (to shut down the Env on the server).
        test.terminate()

    def test_environment_stepper_on_random_env_with_returning_action_probs(self):
        state_space = FloatBox(shape=(2,))
        action_space = IntBox(4)
        preprocessor_spec = [dict(type="multiply", factor=3)]
        neural_network_spec = config_from_path("configs/test_simple_nn.json")
        exploration_spec = None
        actor_component = ActorComponent(
            preprocessor_spec, dict(neural_network=neural_network_spec, action_space=action_space), exploration_spec
        )
        environment_stepper = EnvironmentStepper(
            environment_spec=dict(
                type="random_env", state_space=state_space, action_space=action_space, deterministic=True
            ),
            actor_component_spec=actor_component,
            state_space=state_space,
            reward_space="float32",
            add_action_probs=True,
            action_probs_space=FloatBox(shape=(4,), add_batch_rank=True),
            num_steps=3
        )

        test = ComponentTest(
            component=environment_stepper,
            action_space=action_space,
        )

        # Reset the stepper.
        test.test("reset")

        # Step 3 times through the Env and collect results.
        # 1st return value is the step-op (None), 2nd return value is the tuple of items (3 steps each), with each
        # step containing: Preprocessed state, actions, rewards, episode returns, terminals, (raw) next-states.
        expected_r = np.array([0.19806287, 0.68535984, 0.81262094])
        expected = (None, (
            np.array([[2.313962 , 0.06225585], [1.4955211, 0.67438996], [0.5073325, 0.26501945]]),  # p(s)
            np.array([0, 0, 0]),  # a
            expected_r,  # r
            np.array([expected_r[:1].sum(), expected_r[:2].sum(), expected_r[:3].sum()]),  # episode's accumulated returns
            np.array([False, False, False]),
            np.array([[0.49850702, 0.22479665], [0.16911083, 0.08833981], [0.00394827, 0.51219225]]),  # s' (raw)
            np.array([[0.3300916, 0.1521335, 0.3016182, 0.2161567],
                      [0.4281767, 0.1859898, 0.163224, 0.2226095],
                      [0.311745, 0.2305052, 0.2134148, 0.2443351]])  # action probs
        ))
        test.test("step", expected_outputs=expected)

        # Step again, check whether stitching of states/etc.. works.
        expected_r2 = np.array([0.91777414, 0.37334076, 0.617767])
        expected = (None, (
            np.array([[0.0118448, 1.5365767], [2.165266, 0.87562823], [1.6276331, 0.42651013]]),  # p(s)
            np.array([0, 0, 0]),  # a
            expected_r2,  # r
            np.array([expected_r.sum() + expected_r2[0], expected_r.sum() + expected_r2[:2].sum(), expected_r.sum() + expected_r2[:3].sum()]),
            np.array([False, False, False]),
            np.array([[0.7217553, 0.29187608], [0.54254436, 0.14217004], [0.44183317, 0.434014]]),  # s' (raw)
            np.array([[0.51214, 0.22756, 0.057, 0.2033],
                      [0.49673, 0.1566, 0.14199, 0.20468],
                      [0.3835, 0.18188, 0.20831, 0.22631]])
        ))
        test.test("step", expected_outputs=expected, decimals=5)

        # Make sure we close the session (to shut down the Env on the server).
        test.terminate()

    def test_environment_stepper_on_random_env_with_action_probs_lstm(self):
        state_space = FloatBox(shape=(2,))
        action_space = IntBox(4)
        internal_states_space = Tuple(FloatBox(shape=(10,)), FloatBox(shape=(10,)))
        preprocessor_spec = [dict(type="multiply", factor=3)]
        neural_network_spec = config_from_path("configs/test_lstm_nn.json")
        exploration_spec = None
        actor_component = ActorComponent(
            preprocessor_spec, dict(neural_network=neural_network_spec, action_space=action_space), exploration_spec
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
            action_probs_space=FloatBox(shape=(4,), add_batch_rank=True),
            num_steps=3,
        )

        test = ComponentTest(
            component=environment_stepper,
            action_space=action_space,
        )

        # Reset the stepper.
        test.test("reset")

        # Step 3 times through the Env and collect results.
        # 1st return value is the step-op (None), 2nd return value is the tuple of items (3 steps each), with each
        # step containing: Preprocessed state, actions, rewards, episode returns, terminals, (raw) next-states.
        expected_r = np.array([0.19806287, 0.68535984, 0.81262094])
        expected = (None, (
            np.array([[2.313962 , 0.06225585], [1.4955211, 0.67438996], [0.5073325, 0.26501945]]),  # p(s)
            np.array([0, 0, 0]),  # a
            expected_r,  # r
            np.array([expected_r[:1].sum(), expected_r[:2].sum(), expected_r[:3].sum()]),  # episode's accumulated returns
            np.array([False, False, False]),
            np.array([[0.49850702, 0.22479665], [0.16911083, 0.08833981], [0.00394827, 0.51219225]]),  # s' (raw)
            np.array([[0.3300916, 0.1521335, 0.3016182, 0.2161567],
                      [0.4281767, 0.1859898, 0.163224, 0.2226095],
                      [0.311745, 0.2305052, 0.2134148, 0.2443351]])  # action probs
        ))
        test.test("step", expected_outputs=expected)

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
            dict(neural_network=agent_config["network_spec"],
                 action_adapter_spec=agent_config["action_adapter_spec"],
                 action_space=action_space),
            agent_config["exploration_spec"]
        )
        environment_stepper = EnvironmentStepper(
            environment_spec=environment_spec,
            actor_component_spec=actor_component,
            state_space=state_space,
            reward_space="float",
            num_steps=2000
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
        self.assertTrue(out[0] is None)  # the step op (no_op).
        self.assertTrue(isinstance(out[1], DataOpTuple))  # the step results as a tuple (see below)

        # Check types of single data.
        self.assertTrue(out[1][0].dtype == np.float32)  # preprocessed states
        self.assertTrue(out[1][0].min() >= 0.0)  # make sure we have pixels / 255
        self.assertTrue(out[1][0].max() <= 1.0)
        self.assertTrue(out[1][1].dtype == np.int32)  # actions
        self.assertTrue(out[1][2].dtype == np.float32)  # rewards
        self.assertTrue(out[1][3].dtype == np.float32)  # episode return
        self.assertTrue(out[1][4].dtype == np.bool_)  # next-state is terminal?
        self.assertTrue(out[1][5].dtype == np.uint8)  # next state (raw, not preprocessed)
        self.assertTrue(out[1][5].min() >= 0)  # make sure we have pixels
        self.assertTrue(out[1][5].max() <= 255)

        # Check whether episode returns match single rewards (including resetting after each terminal signal).
        episode_returns = 0.0
        for i in range(environment_stepper.num_steps):
            episode_returns += out[1][2][i]
            self.assertAlmostEqual(episode_returns, out[1][3][i])
            # Terminal: Reset accumulated episode-return before next step.
            if out[1][4][i] is np.bool_(True):
                episode_returns = 0.0

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
            dict(neural_network=agent_config["network_spec"],
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
        time_steps = 2000
        time_start = time.monotonic()
        for i in range(time_steps):
            preprocessed_s, a = test.test(("get_preprocessed_state_and_action", np.array([s])))
            # Act in env.
            s, r, t, _ = dummy_env.step(a[0])  # remove batch
            if t is True:
                s = dummy_env.reset()
        time_end = time.monotonic()
        print("Done running {} steps in bare-metal env in {}sec.".format(time_steps, time_end - time_start))
        test.terminate()
