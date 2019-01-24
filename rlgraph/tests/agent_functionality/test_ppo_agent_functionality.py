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
import numpy as np
import unittest

from rlgraph.agents import Agent, PPOAgent
import rlgraph.spaces as spaces
from rlgraph.components.loss_functions.dqn_loss_function import DQNLossFunction
from rlgraph.environments import GridWorld, RandomEnv, OpenAIGymEnv
from rlgraph.execution.single_threaded_worker import SingleThreadedWorker
from rlgraph.spaces import FloatBox, BoolBox
from rlgraph.tests.test_util import config_from_path
from rlgraph.utils import root_logger, one_hot
from rlgraph.tests.agent_test import AgentTest


class TestPPOAgentFunctionality(unittest.TestCase):
    """
    Tests the PPO Agent's functionality.
    """
    root_logger.setLevel(level=logging.DEBUG)

    def test_post_processing(self):
        """
        Tests external batch post-processing for the PPO agent.
        """
        env = OpenAIGymEnv("Pong-v0", frameskip=4, max_num_noops=30, episodic_life=True)
        agent_config = config_from_path("configs/ppo_agent_for_pong.json")
        agent = PPOAgent.from_spec(
            agent_config,
            state_space=env.state_space,
            action_space=env.action_space
        )
        num_samples = 200
        states = agent.preprocessed_state_space.sample(num_samples)
        reward_space = FloatBox(add_batch_rank=True)
        terminal_space = BoolBox(add_batch_rank=True)
        sequence_indices_space = BoolBox(add_batch_rank=True)

        # GAE is separately tested, just testing if this API method returns results.
        pg_advantages = agent.post_process(dict(
            states=states,
            rewards=reward_space.sample(num_samples),
            terminals=terminal_space.sample(num_samples, fill_value=0),
            sequence_indices=sequence_indices_space.sample(num_samples, fill_value=0)
        ))

    def test_external_update(self):
        """
        Tests updated from post-processed and non-post-processed data.
        """
        pass
