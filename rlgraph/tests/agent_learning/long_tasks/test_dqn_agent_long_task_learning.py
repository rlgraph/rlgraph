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

import unittest
import logging

from rlgraph.environments import OpenAIGymEnv
from rlgraph.agents import Agent
from rlgraph.execution import SingleThreadedWorker
from rlgraph.spaces import FloatBox
from rlgraph.utils import root_logger
from rlgraph.tests.test_util import config_from_path


class TestDQNAgentLongTaskLearning(unittest.TestCase):
    """
    Tests whether the DQNAgent can learn in tough environments.
    """
    root_logger.setLevel(level=logging.INFO)

    pong_preprocessed_state_space = FloatBox(shape=(80, 80, 4), add_batch_rank=True)

    def test_dqn_on_pong(self):
        """
        Creates a DQNAgent and runs it via a Runner on an openAI Pong Env.
        """
        env = OpenAIGymEnv("Pong-v0", frameskip=4, max_num_noops=30, episodic_life=True, visualize=False)
        agent_config = config_from_path("configs/dqn_agent_for_pong.json")
        preprocessing_spec = agent_config.pop("preprocessor_spec")
        agent = Agent.from_spec(
            # Uses 2015 DQN parameters as closely as possible.
            agent_config,
            state_space=self.pong_preprocessed_state_space,
            # Try with "reduced" action space (actually only 3 actions, up, down, no-op)
            action_space=env.action_space
        )

        time_steps = 4000000
        worker = SingleThreadedWorker(
            env_spec=lambda: env, agent=agent, render=True,
            preprocessing_spec=preprocessing_spec,
            worker_executes_preprocessing=True
        )
        results = worker.execute_timesteps(time_steps, use_exploration=True)

        #self.assertEqual(results["timesteps_executed"], time_steps)
        #self.assertEqual(results["env_frames"], time_steps)
        #self.assertAlmostEqual(results["mean_episode_reward"], -2.9207317073170733)
        #self.assertAlmostEqual(results["max_episode_reward"], 0.0)
        #self.assertEqual(results["episodes_executed"], 328)

