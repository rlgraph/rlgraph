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
from rlgraph.spaces import FloatBox
from rlgraph.utils import root_logger
from rlgraph.tests.test_util import config_from_path, recursive_assert_almost_equal


class TestDQNAgentShortTaskLearning(unittest.TestCase):
    """
    Tests whether the DQNAgent can learn in simple environments.
    """
    root_logger.setLevel(level=logging.INFO)
    grid_world_preprocessing_spec = [dict(
        type="reshape",
        flatten=True
    )]

    def test_dqn_on_2x2_grid_world(self):
        """
        Creates a DQNAgent and runs it via a Runner on a simple 2x2 GridWorld.
        """
        dummy_env = GridWorld("2x2")
        agent = DQNAgent.from_spec(
            config_from_path("configs/dqn_agent_for_2x2_grid.json"),
            double_q=False,
            dueling_q=False,
            state_space=FloatBox(shape=(4,), add_batch_rank=True),
            action_space=dummy_env.action_space,
            observe_spec=dict(buffer_size=100),
            execution_spec=dict(seed=12),
            update_spec=dict(update_interval=4, batch_size=24, sync_interval=32),
            optimizer_spec=dict(type="adam", learning_rate=0.05),
            store_last_q_table=True
        )

        time_steps = 1000
        worker = SingleThreadedWorker(
            env_spec=lambda: GridWorld("2x2"),
            agent=agent,
            preprocessing_spec=self.grid_world_preprocessing_spec,
            worker_executes_preprocessing=True
        )
        results = worker.execute_timesteps(time_steps, use_exploration=True)

        print("STATES:\n{}".format(agent.last_q_table["states"]))
        print("\n\nQ(s,a)-VALUES:\n{}".format(np.round_(agent.last_q_table["q_values"], decimals=2)))

        self.assertEqual(results["timesteps_executed"], time_steps)
        self.assertEqual(results["env_frames"], time_steps)
        self.assertGreaterEqual(results["mean_episode_reward"], -3.5)
        self.assertGreaterEqual(results["max_episode_reward"], 0.0)
        self.assertLessEqual(results["episodes_executed"], 350)

        # Check q-table for correct values.
        expected_q_values_per_state = {
            (1.0, 0, 0, 0): (-1, -5, 0, -1),
            (0, 1.0, 0, 0): (-1, 1, 0, 0)
        }
        for state, q_values in zip(agent.last_q_table["states"], agent.last_q_table["q_values"]):
            state, q_values = tuple(state), tuple(q_values)
            assert state in expected_q_values_per_state, \
                "ERROR: state '{}' not expected in q-table as it's a terminal state!".format(state)
            recursive_assert_almost_equal(q_values, expected_q_values_per_state[state], decimals=0)

    def test_double_dqn_on_2x2_grid_world(self):
        """
        Creates a double DQNAgent and runs it via a Runner on a simple 2x2 GridWorld.
        """
        env = GridWorld("2x2")
        agent = DQNAgent.from_spec(
            config_from_path("configs/dqn_agent_for_2x2_grid.json"),
            dueling_q=False,
            state_space=env.state_space,
            action_space=env.action_space,
            observe_spec=dict(buffer_size=100),
            execution_spec=dict(seed=10),
            update_spec=dict(update_interval=4, batch_size=24, sync_interval=32),
            optimizer_spec=dict(type="adam", learning_rate=0.05),
            store_last_q_table=True
        )

        time_steps = 500
        worker = SingleThreadedWorker(env_spec=lambda: env, agent=agent, worker_executes_preprocessing=False)
        results = worker.execute_timesteps(time_steps, use_exploration=True)

        print("STATES:\n{}".format(agent.last_q_table["states"]))
        print("\n\nQ(s,a)-VALUES:\n{}".format(np.round_(agent.last_q_table["q_values"], decimals=2)))

        self.assertEqual(results["timesteps_executed"], time_steps)
        self.assertEqual(results["env_frames"], time_steps)
        self.assertGreaterEqual(results["mean_episode_reward"], -4.5)
        self.assertGreaterEqual(results["max_episode_reward"], 0.0)
        self.assertLessEqual(results["episodes_executed"], 250)

        # Check q-table for correct values.
        expected_q_values_per_state = {
            (1.0, 0, 0, 0): (-1, -5, 0, -1),
            (0, 1.0, 0, 0): (-1, 1, 0, 0)
        }
        for state, q_values in zip(agent.last_q_table["states"], agent.last_q_table["q_values"]):
            state, q_values = tuple(state), tuple(q_values)
            assert state in expected_q_values_per_state, \
                "ERROR: state '{}' not expected in q-table as it's a terminal state!".format(state)
            recursive_assert_almost_equal(q_values, expected_q_values_per_state[state], decimals=0)

    def test_double_dqn_on_4x4_grid_world(self):
        """
        Creates a double DQNAgent and runs it via a Runner on a simple 2x2 GridWorld.
        """
        env = GridWorld("4x4")
        agent = DQNAgent.from_spec(
            config_from_path("configs/dqn_agent_for_4x4_grid.json"),
            dueling_q=False,
            state_space=env.state_space,
            action_space=env.action_space,
            observe_spec=dict(buffer_size=100),
            execution_spec=dict(seed=10),
            update_spec=dict(update_interval=4, batch_size=32, sync_interval=32),
            optimizer_spec=dict(type="adam", learning_rate=0.005),
            store_last_q_table=True
        )

        time_steps = 3000
        worker = SingleThreadedWorker(env_spec=lambda: env, agent=agent, worker_executes_preprocessing=False)
        results = worker.execute_timesteps(time_steps, use_exploration=True)

        print("STATES:\n{}".format(agent.last_q_table["states"]))
        print("\n\nQ(s,a)-VALUES:\n{}".format(np.round_(agent.last_q_table["q_values"], decimals=2)))

        self.assertEqual(results["timesteps_executed"], time_steps)
        self.assertEqual(results["env_frames"], time_steps)
        self.assertGreaterEqual(results["mean_episode_reward"], -10)
        self.assertGreaterEqual(results["max_episode_reward"], -4)
        self.assertLessEqual(results["episodes_executed"], 1000)

        # Check q-table for correct values.
        expected_q_values_per_state = {
            (1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0): (-5, -4, -4, -4),  # 0
            (0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0): (-5, -5, -3, -4),  # 1
            (0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0): (-4, -2, -5, -3),  # 2
            # 3=terminal
            (0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0): (-4, -3, -5, -5),  # 4
            # 5=terminal
            (0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0): (-5, -1, -1, -3),  # 6
            (0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0): (-2, 0, -1, -5),  # 7
            (0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0): (-3, -4, -2, -4),  # 8
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0): (-3, -5, -1, -5),  # 9
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0): (-2, -5, 0, -2),  # 10
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0): (-1, 1, 0, -1),  # 11
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0): (-4, -4, -5, -3),  # 12
            # 13=terminal
            # 14=terminal
            # 15=terminal
        }
        for state, q_values in zip(agent.last_q_table["states"], agent.last_q_table["q_values"]):
            state, q_values = tuple(state), tuple(q_values)
            assert state in expected_q_values_per_state, \
                "ERROR: state '{}' not expected in q-table as it's a terminal state!".format(state)
            recursive_assert_almost_equal(q_values, expected_q_values_per_state[state], decimals=0)

    def test_dqn_on_cart_pole(self):
        """
        Creates a DQNAgent and runs it via a Runner on the CartPole Env.
        """
        env = OpenAIGymEnv("CartPole-v0")
        env.seed(10)
        agent = DQNAgent.from_spec(
            config_from_path("configs/dqn_agent_for_cartpole.json"),
            double_q=False,
            dueling_q=False,
            state_space=env.state_space,
            action_space=env.action_space,
            observe_spec=dict(buffer_size=200),
            execution_spec=dict(seed=15),
            update_spec=dict(update_interval=4, batch_size=24, sync_interval=64),
            optimizer_spec=dict(type="adam", learning_rate=0.05),
            store_last_q_table=True
        )

        time_steps = 5000
        worker = SingleThreadedWorker(env_spec=lambda : env, agent=agent, render=False,
                                      worker_executes_preprocessing=False)
        results = worker.execute_timesteps(time_steps, use_exploration=True)

        #print("STATES:\n{}".format(agent.last_q_table["states"]))
        #print("\n\nQ(s,a)-VALUES:\n{}".format(np.round_(agent.last_q_table["q_values"], decimals=2)))

        self.assertEqual(results["timesteps_executed"], time_steps)
        self.assertEqual(results["env_frames"], time_steps)
        self.assertGreaterEqual(results["mean_episode_reward"], 20)
        self.assertGreaterEqual(results["max_episode_reward"], 100.0)
        self.assertLessEqual(results["episodes_executed"], 200)

    def test_double_dueling_dqn_on_cart_pole(self):
        """
        Creates a double and dueling DQNAgent and runs it via a Runner on the CartPole Env.
        """
        env = OpenAIGymEnv("CartPole-v0")
        env.seed(10)
        agent = DQNAgent.from_spec(
            config_from_path("configs/dqn_agent_for_cartpole.json"),
            double_q=True,
            dueling_q=True,
            state_space=env.state_space,
            action_space=env.action_space,
            observe_spec=dict(buffer_size=200),
            execution_spec=dict(seed=156),
            update_spec=dict(update_interval=4, batch_size=64, sync_interval=16),
            optimizer_spec=dict(type="adam", learning_rate=0.05),
            store_last_q_table=True
        )

        time_steps = 3000
        worker = SingleThreadedWorker(env_spec=lambda : env, agent=agent, render=False,
                                      worker_executes_preprocessing=False)
        results = worker.execute_timesteps(time_steps, use_exploration=True)

        #print("STATES:\n{}".format(agent.last_q_table["states"]))
        #print("\n\nQ(s,a)-VALUES:\n{}".format(np.round_(agent.last_q_table["q_values"], decimals=2)))

        self.assertEqual(results["timesteps_executed"], time_steps)
        self.assertEqual(results["env_frames"], time_steps)
        self.assertGreaterEqual(results["mean_episode_reward"], 25)
        self.assertGreaterEqual(results["max_episode_reward"], 200.0)
        self.assertLessEqual(results["episodes_executed"], 200)

