# Copyright 2018 The RLgraph authors. All Rights Reserved.
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
import unittest
import logging

from rlgraph.environments import GridWorld, OpenAIGymEnv
from rlgraph.agents import DQNAgent
from rlgraph.execution import SingleThreadedWorker
from rlgraph.utils import root_logger


class TestDQNAgentShortTaskLearning(unittest.TestCase):
    """
    Tests whether the DQNAgent can learn in simple environments.
    """
    root_logger.setLevel(level=logging.INFO)

    def test_dqn_on_2x2_grid_world(self):
        """
        Creates a DQNAgent and runs it via a Runner on a simple 2x2 GridWorld.
        """
        env = GridWorld("2x2")
        agent = DQNAgent.from_spec(
            "../configs/dqn_agent_for_2x2_grid.json",
            #discount=1.0,  # very short episodes -> no discount.
            double_q=False,
            dueling_q=False,
            state_space=env.state_space,
            action_space=env.action_space,
            observe_spec=dict(buffer_size=100),
            execution_spec=dict(seed=10),
            update_spec=dict(update_interval=4, batch_size=24, sync_interval=32),
            optimizer_spec=dict(learning_rate=0.05),
            store_last_q_table=True
        )

        time_steps = 1000
        worker = SingleThreadedWorker(environment=env, agent=agent)
        results = worker.execute_timesteps(time_steps, use_exploration=True)

        print("STATES:\n{}".format(agent.last_q_table["states"]))
        print("\n\nQ(s,a)-VALUES:\n{}".format(np.round_(agent.last_q_table["q_values"], decimals=2)))

        self.assertEqual(results["timesteps_executed"], time_steps)
        self.assertEqual(results["env_frames"], time_steps)
        self.assertAlmostEqual(results["mean_episode_reward"], -2.92, places=2)
        self.assertAlmostEqual(results["max_episode_reward"], 0.0)
        self.assertAlmostEqual(results["final_episode_reward"], -1)
        self.assertEqual(results["episodes_executed"], 328)

    def test_double_dqn_on_2x2_grid_world(self):
        """
        Creates a double DQNAgent and runs it via a Runner on a simple 2x2 GridWorld.
        """
        env = GridWorld("2x2")
        agent = DQNAgent.from_spec(
            "../configs/dqn_agent_for_2x2_grid.json",
            dueling_q=False,
            state_space=env.state_space,
            action_space=env.action_space,
            observe_spec=dict(buffer_size=100),
            execution_spec=dict(seed=10),
            update_spec=dict(update_interval=4, batch_size=24, sync_interval=32),
            optimizer_spec=dict(learning_rate=0.05),
            store_last_q_table=True
        )

        time_steps = 500
        worker = SingleThreadedWorker(environment=env, agent=agent)
        results = worker.execute_timesteps(time_steps, use_exploration=True)

        self.assertEqual(results["timesteps_executed"], time_steps)
        self.assertEqual(results["env_frames"], time_steps)
        self.assertAlmostEqual(results["mean_episode_reward"], -4.160583941605839)
        self.assertAlmostEqual(results["max_episode_reward"], 0.0)
        self.assertAlmostEqual(results["final_episode_reward"], -1)
        self.assertEqual(results["episodes_executed"], 137)

    def test_double_dueling_dqn_on_4x4_grid_world(self):
        """
        Creates a double DQNAgent and runs it via a Runner on a simple 2x2 GridWorld.
        """
        env = GridWorld("4x4")
        agent = DQNAgent.from_spec(
            "../configs/dqn_agent_for_4x4_grid.json",
            state_space=env.state_space,
            action_space=env.action_space,
            observe_spec=dict(buffer_size=100),
            execution_spec=dict(seed=10),
            update_spec=dict(update_interval=4, batch_size=32, sync_interval=32),
            optimizer_spec=dict(learning_rate=0.1),
            store_last_q_table=True
        )

        time_steps = 3000
        worker = SingleThreadedWorker(environment=env, agent=agent)
        results = worker.execute_timesteps(time_steps, use_exploration=True)

        print("STATES:\n{}".format(agent.last_q_table["states"]))
        print("\n\nQ(s,a)-VALUES:\n{}".format(np.round_(agent.last_q_table["q_values"], decimals=2)))

        self.assertEqual(results["timesteps_executed"], time_steps)
        self.assertEqual(results["env_frames"], time_steps)
        self.assertAlmostEqual(results["mean_episode_reward"], -8, places=0)
        self.assertAlmostEqual(results["max_episode_reward"], -4)
        self.assertAlmostEqual(results["final_episode_reward"], -5)
        self.assertEqual(results["episodes_executed"], 551)

    def test_dqn_on_cart_pole(self):
        """
        Creates a DQNAgent and runs it via a Runner on the CartPole Env.
        """
        env = OpenAIGymEnv("CartPole-v0")
        env.seed(10)
        agent = DQNAgent.from_spec(
            "../configs/dqn_agent_for_cartpole.json",
            double_q=False,
            dueling_q=False,
            state_space=env.state_space,
            action_space=env.action_space,
            observe_spec=dict(buffer_size=200),
            execution_spec=dict(seed=10),
            update_spec=dict(update_interval=4, batch_size=24, sync_interval=64),
            optimizer_spec=dict(learning_rate=0.0002),
            store_last_q_table=True
        )

        time_steps = 5000
        worker = SingleThreadedWorker(environment=env, agent=agent, render=True)
        results = worker.execute_timesteps(time_steps, use_exploration=True)

        print("STATES:\n{}".format(agent.last_q_table["states"]))
        print("\n\nQ(s,a)-VALUES:\n{}".format(np.round_(agent.last_q_table["q_values"], decimals=2)))

        self.assertEqual(results["timesteps_executed"], time_steps)
        self.assertEqual(results["env_frames"], time_steps)
        self.assertAlmostEqual(results["mean_episode_reward"], 87.71929824561404)
        self.assertAlmostEqual(results["max_episode_reward"], 200.0)
        self.assertAlmostEqual(results["final_episode_reward"], 49.0)
        self.assertEqual(results["episodes_executed"], 57)

    def test_double_dueling_dqn_on_cart_pole(self):
        """
        Creates a double and dueling DQNAgent and runs it via a Runner on the CartPole Env.
        """
        env = OpenAIGymEnv("CartPole-v0")
        env.seed(10)
        agent = DQNAgent.from_spec(
            "../configs/dqn_agent_for_cartpole.json",
            double_q=True,
            dueling_q=True,
            state_space=env.state_space,
            action_space=env.action_space,
            observe_spec=dict(buffer_size=200),
            execution_spec=dict(seed=10),
            update_spec=dict(update_interval=4, batch_size=64, sync_interval=16),
            optimizer_spec=dict(type="adam", learning_rate=0.02),
            store_last_q_table=True
        )

        time_steps = 3000
        worker = SingleThreadedWorker(environment=env, agent=agent, render=True)
        results = worker.execute_timesteps(time_steps, use_exploration=True)

        print("STATES:\n{}".format(agent.last_q_table["states"]))
        print("\n\nQ(s,a)-VALUES:\n{}".format(np.round_(agent.last_q_table["q_values"], decimals=2)))

        self.assertEqual(results["timesteps_executed"], time_steps)
        self.assertEqual(results["env_frames"], time_steps)
        self.assertAlmostEqual(results["mean_episode_reward"], 58.8235294117647)
        self.assertAlmostEqual(results["max_episode_reward"], 200.0)
        self.assertAlmostEqual(results["final_episode_reward"], 60.0)
        self.assertEqual(results["episodes_executed"], 51)

