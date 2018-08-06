# Copyright 2018 The YARL-Project, All Rights Reserved.
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

import copy
import numpy as np
import unittest

from rlgraph.environments.environment import Environment
from rlgraph.components.neural_networks.actor_component import ActorComponent
from rlgraph.components.common.environment_stepper import EnvironmentStepper
from rlgraph.spaces import FloatBox, IntBox
from rlgraph.tests import ComponentTest
from rlgraph.tests.test_util import agent_config_from_path


class TestEnvironmentStepper(unittest.TestCase):
    """
    Tests for the EnvironmentStepper Component using a simple RandomEnv.
    """
    def test_environment_stepper_on_random_env(self):
        state_space = FloatBox(shape=(1,))
        action_space = IntBox(2)
        preprocessor_spec = None
        neural_network_spec = "../configs/test_simple_nn.json"
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
            reward_space="float32"
        )

        test = ComponentTest(component=environment_stepper, action_space=action_space)

        # Reset the stepper.
        test.test("reset")

        # Step 3 times through the Env and collect results.
        # 1st return value is the step-op (None), 2nd return value is the tuple of items (3 steps each), with each
        # step containing: Preprocessed state, actions, rewards, episode returns, terminals, (raw) next-states.
        expected = (None, (
            np.array([[0.77132064], [0.74880385], [0.19806287]]),  # p(s)
            np.array([0, 0, 0]),  # a
            np.array([0.49850702, 0.7605307, 0.68535984]),  # r
            np.array([0.49850702, 1.2590377, 1.9443976]),  # episode's accumulated returns
            np.array([False, False, False]),
            np.array([[0.74880385], [0.19806287], [0.08833981]]),  # s' (raw)
        ))
        test.test(("step", 3), expected_outputs=expected)

    def test_environment_stepper_on_pong(self):
        environment_spec = dict(type="openai_gym", gym_env="Pong-v0", frameskip=4)
        dummy_env = Environment.from_spec(copy.deepcopy(environment_spec))
        state_space = dummy_env.state_space
        action_space = dummy_env.action_space
        agent_config = agent_config_from_path("../configs/dqn_agent_for_pong.json")
        actor_component = ActorComponent(
            agent_config["preprocessing_spec"],
            dict(neural_network=agent_config["network_spec"], action_space=action_space),
            agent_config["exploration_spec"]
        )
        environment_stepper = EnvironmentStepper(
            environment_spec=environment_spec,
            actor_component_spec=actor_component,
            state_space=state_space,
            reward_space="float64"
        )

        test = ComponentTest(component=environment_stepper, action_space=action_space)

        # Reset the stepper.
        test.test("reset")

        # Step 30 times through the Env and collect results.
        # 1st return value is the step-op (None), 2nd return value is the tuple of items (3 steps each), with each
        # step containing: Preprocessed state, actions, rewards, episode returns, terminals, (raw) next-states.
        # TODO: Fill in values for certain frames.
        #expected = (None, (
        #    np.array([[0.77132064], [0.74880385], [0.19806287]]),  # p(s)
        #    np.array([0, 0, 0]),  # a
        #    np.array([0.49850702, 0.7605307, 0.68535984]),  # r
        #    np.array([0.49850702, 1.2590377, 1.9443976]),  # episode's accumulated returns
        #    np.array([False, False, False]),
        #    np.array([[0.74880385], [0.19806287], [0.08833981]]),  # s' (raw)
        #))

        test.test(("step", [3, 0]), expected_outputs=None)
