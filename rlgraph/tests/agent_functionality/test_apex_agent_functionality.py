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
from rlgraph.agents import Agent
from rlgraph.environments import OpenAIGymEnv
from rlgraph.tests import recursive_assert_almost_equal
from rlgraph.tests.test_util import config_from_path
from rlgraph.utils import root_logger

import numpy as np


class TestApexAgentFunctionality(unittest.TestCase):
    """
    Tests Ape-X specific functionality.
    """
    root_logger.setLevel(level=logging.INFO)

    def test_apex_weight_syncing(self):
        agent_config = config_from_path("configs/ray_apex_for_pong.json")
        agent_config["execution_spec"].pop("ray_spec")
        environment = OpenAIGymEnv("Pong-v0", frameskip=4)

        agent = Agent.from_spec(
            agent_config,
            state_space=environment.state_space,
            action_space=environment.action_space
        )

        weights = agent.get_weights()["policy_weights"]
        print("type weights = ", type(weights))
        for variable, value in weights.items():
            print("Type value = ", type(value))
            value += 0.01
        agent.set_weights(weights)

        new_weights = agent.get_weights()["policy_weights"]
        recursive_assert_almost_equal(weights, new_weights)

    def test_update_from_external(self):
        agent_config = config_from_path("configs/ray_apex_for_pong.json")
        agent_config["execution_spec"].pop("ray_spec")
        environment = OpenAIGymEnv("Pong-v0", frameskip=4)

        agent = Agent.from_spec(
            agent_config,
            state_space=environment.state_space,
            action_space=environment.action_space
        )

        batch = {
            "states": agent.preprocessed_state_space.sample(200),
            "actions": environment.action_space.sample(200),
            "rewards": np.zeros(200, dtype=np.float32),
            "terminals": [False] * 200,
            "next_states": agent.preprocessed_state_space.sample(200),
            "importance_weights":  np.ones(200, dtype=np.float32)
        }

        agent.update(batch)
