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

import logging
import unittest

from rlgraph.agents import Agent, PPOAgent
from rlgraph.environments import GridWorld, OpenAIGymEnv
from rlgraph.tests.test_util import config_from_path, recursive_assert_almost_equal
from rlgraph.utils import root_logger


class TestBaseAgentFunctionality(unittest.TestCase):
    """
    Tests the base Agent's functionality.
    """
    root_logger.setLevel(level=logging.DEBUG)

    def test_weights_getting_setting(self):
        """
        Tests getting and setting of the Agent's weights.
        """
        env = GridWorld(world="2x2")
        agent = Agent.from_spec(
            config_from_path("configs/dqn_agent_for_functionality_test.json"),
            state_space=env.state_space,
            action_space=env.action_space
        )

        weights = agent.get_weights()
        new_weights = {}
        for key, weight in weights["policy_weights"].items():
            new_weights[key] = weight + 0.01

        agent.set_weights(new_weights)
        new_actual_weights = agent.get_weights()

        recursive_assert_almost_equal(new_actual_weights["policy_weights"], new_weights)

    def test_value_function_weights(self):
        """
        Tests changing of value function weights.
        """
        env = OpenAIGymEnv("Pong-v0")
        agent_config = config_from_path("configs/ppo_agent_for_pong.json")
        agent = PPOAgent.from_spec(
            agent_config,
            state_space=env.state_space,
            action_space=env.action_space
        )
        weights = agent.get_weights()
        assert "value_function_weights" in weights
        assert "policy_weights" in weights

        policy_weights = weights["policy_weights"]
        value_function_weights = weights["value_function_weights"]

        # Just change vf weights.
        for key, weight in value_function_weights.items():
            value_function_weights[key] = weight + 0.01
        agent.set_weights(policy_weights, value_function_weights)
        new_actual_weights = agent.get_weights()

        recursive_assert_almost_equal(new_actual_weights["value_function_weights"],
                                      value_function_weights)
