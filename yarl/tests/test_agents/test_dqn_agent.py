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

import logging
import unittest

from yarl.agents import DQNAgent
import yarl.spaces as spaces
from yarl.envs import GridWorld, RandomEnv, OpenAIGymEnv
from yarl.execution.single_threaded_worker import SingleThreadedWorker
from yarl.utils import root_logger


class TestDQNAgent(unittest.TestCase):
    """
    Tests the DQN Agent on simple deterministic learning tasks.
    """
    root_logger.setLevel(level=logging.INFO)

    def test_dqn_assembly(self):
        """
        Creates a DQNAgent and runs it for a few steps in the random Env.
        """
        env = RandomEnv(state_space=spaces.IntBox(2), action_space=spaces.IntBox(2), deterministic=True)
        agent = DQNAgent.from_spec(
            "configs/test_dqn_agent_for_random_env.json",
            state_space=env.state_space,
            action_space=env.action_space
        )

        worker = SingleThreadedWorker(environment=env, agent=agent)
        results = worker.execute_timesteps(1000, deterministic=True)

        self.assertEqual(results["timesteps_executed"], 1000)
        self.assertEqual(results["env_frames"], 1000)
        # Assert deterministic execution of Env and Agent.
        self.assertAlmostEqual(results["mean_episode_reward"], 4.607321286981477)
        self.assertAlmostEqual(results["max_episode_reward"], 24.909519721955455)
        self.assertAlmostEqual(results["final_episode_reward"], 1.3333066872744532)

    def test_dqn_on_2x2_grid_world(self):
        """
        Creates a DQNAgent and runs it via a Runner on a simple 2x2 GridWorld.
        """
        env = GridWorld("2x2")
        agent = DQNAgent.from_spec(
            "configs/test_dqn_agent_for_2x2_grid.json",
            state_space=env.state_space,
            action_space=env.action_space,
            observe_spec=dict(buffer_size=100),
            execution_spec=dict(seed=10),
            update_spec=dict(update_interval=4, batch_size=8, sync_interval=32),
            optimizer_spec=dict(learning_rate=0.01)
        )

        worker = SingleThreadedWorker(environment=env, agent=agent)
        results = worker.execute_timesteps(1000, deterministic=True)

        self.assertEqual(results["timesteps_executed"], 1000)
        self.assertEqual(results["env_frames"], 1000)
        self.assertAlmostEqual(results["mean_episode_reward"], -2.8220858895705523)
        self.assertAlmostEqual(results["max_episode_reward"], 0.0)
        self.assertAlmostEqual(results["final_episode_reward"], -1)
        self.assertEqual(results["episodes_executed"], 326)

    def test_dqn_on_cart_pole(self):
        """
        Creates a DQNAgent and runs it via a Runner on the CartPole Env.
        """
        env = OpenAIGymEnv("CartPole-v0")
        env.seed(10)
        agent = DQNAgent.from_spec(
            "configs/test_dqn_agent_for_cartpole.json",
            state_space=env.state_space,
            action_space=env.action_space,
            observe_spec=dict(buffer_size=100),
            execution_spec=dict(seed=10),
            update_spec=dict(update_interval=4, batch_size=16, sync_interval=32),
            optimizer_spec=dict(learning_rate=0.002)
        )

        worker = SingleThreadedWorker(environment=env, agent=agent)
        results = worker.execute_timesteps(10000, deterministic=True)

        self.assertEqual(results["timesteps_executed"], 10000)
        self.assertEqual(results["env_frames"], 10000)
        self.assertAlmostEqual(results["mean_episode_reward"], 82.64462809917356)
        self.assertAlmostEqual(results["max_episode_reward"], 200.0)
        self.assertAlmostEqual(results["final_episode_reward"], 43.0)
        self.assertEqual(results["episodes_executed"], 121)
