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

import unittest
import logging

from yarl.envs import GridWorld, OpenAIGymEnv
from yarl.agents import DQNAgent
from yarl.execution import SingleThreadedWorker
from yarl.utils import root_logger


class TestDQNAgentAssembly(unittest.TestCase):
    """
    Tests the DQN Agent assembly on the random Env.
    """
    root_logger.setLevel(level=logging.INFO)

    def test_dqn_on_2x2_grid_world(self):
        """
        Creates a DQNAgent and runs it via a Runner on a simple 2x2 GridWorld.
        """
        env = GridWorld("2x2")
        agent = DQNAgent.from_spec(
            "configs/test_dqn_agent_for_2x2_grid.json",
            double_q=False,
            dueling_q=False,
            state_space=env.state_space,
            action_space=env.action_space,
            observe_spec=dict(buffer_size=100),
            execution_spec=dict(seed=10),
            update_spec=dict(update_interval=4, batch_size=24, sync_interval=32),
            optimizer_spec=dict(learning_rate=0.05)
        )

        time_steps = 10000
        worker = SingleThreadedWorker(environment=env, agent=agent)
        results = worker.execute_timesteps(time_steps, deterministic=True)

        self.assertEqual(results["timesteps_executed"], time_steps)
        self.assertEqual(results["env_frames"], time_steps)
        self.assertAlmostEqual(results["mean_episode_reward"], -2.6261398176291793)
        self.assertAlmostEqual(results["max_episode_reward"], 0.0)
        self.assertAlmostEqual(results["final_episode_reward"], -1)
        self.assertEqual(results["episodes_executed"], 329)

    def test_double_dqn_on_2x2_grid_world(self):
        """
        Creates a double DQNAgent and runs it via a Runner on a simple 2x2 GridWorld.
        """
        env = GridWorld("2x2")
        agent = DQNAgent.from_spec(
            "configs/test_dqn_agent_for_2x2_grid.json",
            dueling_q=False,
            state_space=env.state_space,
            action_space=env.action_space,
            observe_spec=dict(buffer_size=100),
            execution_spec=dict(seed=10),
            update_spec=dict(update_interval=4, batch_size=24, sync_interval=32),
            optimizer_spec=dict(learning_rate=0.05)
        )

        worker = SingleThreadedWorker(environment=env, agent=agent)
        results = worker.execute_timesteps(1000, deterministic=True)

        self.assertEqual(results["timesteps_executed"], 1000)
        self.assertEqual(results["env_frames"], 1000)
        self.assertAlmostEqual(results["mean_episode_reward"], -2.567073170731707)
        self.assertAlmostEqual(results["max_episode_reward"], 0.0)
        self.assertAlmostEqual(results["final_episode_reward"], 0)
        self.assertEqual(results["episodes_executed"], 328)

    def test_double_dueling_dqn_on_4x4_grid_world(self):
        """
        Creates a double DQNAgent and runs it via a Runner on a simple 2x2 GridWorld.
        """
        env = GridWorld("4x4")
        agent = DQNAgent.from_spec(
            "configs/test_dqn_agent_for_4x4_grid.json",
            state_space=env.state_space,
            action_space=env.action_space,
            observe_spec=dict(buffer_size=100),
            execution_spec=dict(seed=10),
            update_spec=dict(update_interval=4, batch_size=24, sync_interval=32),
            optimizer_spec=dict(learning_rate=0.005)
        )

        worker = SingleThreadedWorker(environment=env, agent=agent)
        results = worker.execute_timesteps(10000, deterministic=True)

        self.assertEqual(results["timesteps_executed"], 10000)
        self.assertEqual(results["env_frames"], 10000)
        self.assertAlmostEqual(results["mean_episode_reward"], -26.35128805620609)
        self.assertAlmostEqual(results["max_episode_reward"], -3)
        self.assertAlmostEqual(results["final_episode_reward"], -3)
        self.assertEqual(results["episodes_executed"], 1281)

    def test_dqn_on_cart_pole(self):
        """
        Creates a DQNAgent and runs it via a Runner on the CartPole Env.
        """
        env = OpenAIGymEnv("CartPole-v0")
        env.seed(10)
        agent = DQNAgent.from_spec(
            "configs/test_dqn_agent_for_cartpole.json",
            double_q=False,
            dueling_q=False,
            state_space=env.state_space,
            action_space=env.action_space,
            observe_spec=dict(buffer_size=100),
            execution_spec=dict(seed=10),
            #update_spec=dict(update_interval=4),
            #optimizer_spec=dict(type="adagrad", learning_rate=0.001)
        )

        worker = SingleThreadedWorker(environment=env, agent=agent, render=False)
        results = worker.execute_timesteps(100000, deterministic=True)

        self.assertEqual(results["timesteps_executed"], 10000)
        self.assertEqual(results["env_frames"], 10000)
        self.assertAlmostEqual(results["mean_episode_reward"], 94.33962264150944)
        self.assertAlmostEqual(results["max_episode_reward"], 200.0)
        self.assertAlmostEqual(results["final_episode_reward"], 156.0)
        self.assertEqual(results["episodes_executed"], 106)

    def test_double_dueling_dqn_on_cart_pole(self):
        """
        Creates a double and dueling DQNAgent and runs it via a Runner on the CartPole Env.
        """
        env = OpenAIGymEnv("CartPole-v0")
        env.seed(10)
        agent = DQNAgent.from_spec(
            "configs/test_dqn_agent_for_cartpole.json",
            double_q=False,
            dueling_q=True,
            state_space=env.state_space,
            action_space=env.action_space,
            observe_spec=dict(buffer_size=100),
            execution_spec=dict(seed=10),
            update_spec=dict(update_interval=4, batch_size=16, sync_interval=32),
            optimizer_spec=dict(learning_rate=0.005)
        )

        num_timesteps = 10000
        worker = SingleThreadedWorker(environment=env, agent=agent)
        results = worker.execute_timesteps(num_timesteps, deterministic=True)

        print(results)
        self.assertEqual(results["timesteps_executed"], num_timesteps)
        self.assertEqual(results["env_frames"], num_timesteps)
        #self.assertAlmostEqual(results["mean_episode_reward"], 94.33962264150944)
        self.assertAlmostEqual(results["max_episode_reward"], 200.0)
        #self.assertAlmostEqual(results["final_episode_reward"], 156.0)
        #self.assertEqual(results["episodes_executed"], 106)

