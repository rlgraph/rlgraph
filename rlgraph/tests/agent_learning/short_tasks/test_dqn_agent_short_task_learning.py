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
import os
import unittest

import numpy as np

from rlgraph.agents import DQNAgent
from rlgraph.environments import GridWorld, OpenAIGymEnv
from rlgraph.execution import SingleThreadedWorker
from rlgraph.spaces import FloatBox, IntBox
from rlgraph.tests.test_util import config_from_path, recursive_assert_almost_equal
from rlgraph.utils import root_logger
from rlgraph.utils.numpy import one_hot


class TestDQNAgentShortTaskLearning(unittest.TestCase):
    """
    Tests whether the DQNAgent can learn in simple environments.
    """
    root_logger.setLevel(level=logging.INFO)
    # Preprocessed state spaces.
    grid_world_2x2_flattened_state_space = FloatBox(shape=(4,), add_batch_rank=True)
    grid_world_4x4_flattened_state_space = FloatBox(shape=(16,), add_batch_rank=True)
    is_windows = os.name == "nt"

    def test_dqn_on_2x2_grid_world(self):
        """
        Creates a DQNAgent and runs it via a Runner on a simple 2x2 GridWorld.
        """
        dummy_env = GridWorld("2x2")
        agent_config = config_from_path("configs/dqn_agent_for_2x2_gridworld.json")
        preprocessing_spec = agent_config.pop("preprocessing_spec")
        agent = DQNAgent.from_spec(
            agent_config,
            double_q=False,
            dueling_q=False,
            state_space=self.grid_world_2x2_flattened_state_space,
            action_space=dummy_env.action_space,
            execution_spec=dict(seed=12),
            update_spec=dict(update_interval=4, batch_size=24, sync_interval=32),
            optimizer_spec=dict(type="adam", learning_rate=0.05)
        )

        time_steps = 2000
        worker = SingleThreadedWorker(
            env_spec=lambda: GridWorld("2x2"),
            agent=agent,
            preprocessing_spec=preprocessing_spec,
            worker_executes_preprocessing=True
        )
        results = worker.execute_timesteps(time_steps, use_exploration=True)

        self.assertEqual(results["timesteps_executed"], time_steps)
        self.assertEqual(results["env_frames"], time_steps)
        self.assertGreaterEqual(results["mean_episode_reward"], -1.5)
        self.assertGreaterEqual(results["max_episode_reward"], 0.0)
        self.assertLessEqual(results["episodes_executed"], time_steps / 2)

        # Check all learnt Q-values.
        q_values = agent.graph_executor.execute(("get_q_values", one_hot(np.array([0, 1]), depth=4)))[:]
        recursive_assert_almost_equal(q_values[0], (0.8, -5, 0.9, 0.8), decimals=1)
        recursive_assert_almost_equal(q_values[1], (0.8, 1.0, 0.9, 0.9), decimals=1)

    def test_double_dqn_on_2x2_grid_world(self):
        """
        Creates a double DQNAgent and runs it via a Runner on a simple 2x2 GridWorld.
        """
        env_spec = dict(world="2x2")
        dummy_env = GridWorld.from_spec(env_spec)
        agent_config = config_from_path("configs/dqn_agent_for_2x2_gridworld.json")
        preprocessing_spec = agent_config.pop("preprocessing_spec")
        agent = DQNAgent.from_spec(
            agent_config,
            dueling_q=False,
            state_space=self.grid_world_2x2_flattened_state_space,
            action_space=dummy_env.action_space,
            execution_spec=dict(seed=10),
            update_spec=dict(update_interval=4, batch_size=24, sync_interval=32),
            optimizer_spec=dict(type="adam", learning_rate=0.05)
        )

        time_steps = 2000
        worker = SingleThreadedWorker(
            env_spec=lambda: GridWorld.from_spec(env_spec),
            agent=agent,
            preprocessing_spec=preprocessing_spec,
            worker_executes_preprocessing=True
        )
        results = worker.execute_timesteps(time_steps, use_exploration=True)

        self.assertEqual(results["timesteps_executed"], time_steps)
        self.assertEqual(results["env_frames"], time_steps)
        self.assertGreaterEqual(results["mean_episode_reward"], -1.5)
        self.assertGreaterEqual(results["max_episode_reward"], 0.0)
        self.assertLessEqual(results["episodes_executed"], time_steps / 2)

        # Check all learnt Q-values.
        q_values = agent.graph_executor.execute(("get_q_values", one_hot(np.array([0, 1]), depth=4)))[:]
        recursive_assert_almost_equal(q_values[0], (0.8, -5, 0.9, 0.8), decimals=1)
        recursive_assert_almost_equal(q_values[1], (0.8, 1.0, 0.9, 0.9), decimals=1)

    def test_double_dqn_on_2x2_grid_world_with_container_actions(self):
        """
        Creates a double DQNAgent and runs it via a Runner on a simple 2x2 GridWorld using container actions.
        """
        # ftj = forward + turn + jump
        env_spec = dict(world="2x2", action_type="ftj", state_representation="xy+orientation")
        dummy_env = GridWorld.from_spec(env_spec)
        agent_config = config_from_path("configs/dqn_agent_for_2x2_gridworld_with_container_actions.json")
        preprocessing_spec = agent_config.pop("preprocessing_spec")

        agent = DQNAgent.from_spec(
            agent_config,
            double_q=True,
            dueling_q=False,
            state_space=FloatBox(shape=(4,)),
            action_space=dummy_env.action_space,
            execution_spec=dict(seed=15)
        )

        time_steps = 10000
        worker = SingleThreadedWorker(
            env_spec=lambda: GridWorld.from_spec(env_spec),
            agent=agent,
            preprocessing_spec=preprocessing_spec,
            worker_executes_preprocessing=True,
            render=False,
            episode_finish_callback=lambda episode_return, duration, timesteps, env_num:
            print("episode return {}; steps={}".format(episode_return, timesteps))
        )
        results = worker.execute_timesteps(time_steps, use_exploration=True)

        self.assertEqual(results["timesteps_executed"], time_steps)
        self.assertEqual(results["env_frames"], time_steps)
        self.assertGreaterEqual(results["mean_episode_reward"], -2.0)
        self.assertGreaterEqual(results["max_episode_reward"], -1.0)
        self.assertLessEqual(results["episodes_executed"], time_steps / 3)

        # Check q-table for correct values.
        expected_q_values_per_state = {
            (0., 0., -1., 0.): {"forward": (-5.0, -1.0, -1.0), "jump": (0.0, -1.0)},  # start <
            (0., 0., 1., 0.): {"forward": (0.0, -1.0, -1.0), "jump": (0.0, -1.0)},    # start >
            (0., 0., 0., -1.): {"forward": (0.0, -1.0, -1.0), "jump": (0.0, -1.0)},   # start v
            (0., 0., 0., 1.): {"forward": (0.0, -1.0, -1.0), "jump": (0.0, -1.0)},    # start ^
            (0., 1., -1., 0.): {"forward": (0.0, -1.0, -1.0), "jump": (0.0, -1.0)},
            (0., 1., 1., 0.): {"forward": (0.0, -1.0, -1.0), "jump": (0.0, -1.0)},
            (0., 1., 0., -1.): {"forward": (0.0, -1.0, -1.0), "jump": (0.0, -1.0)},
            (0., 1., 0., 1.): {"forward": (0.0, -1.0, -1.0), "jump": (0.0, -1.0)},
        }
        for state, q_values_forward, q_values_jump in zip(
                agent.last_q_table["states"], agent.last_q_table["q_values"]["forward"],
                agent.last_q_table["q_values"]["jump"]
        ):
            state, q_values_forward, q_values_jump = tuple(state), tuple(q_values_forward), tuple(q_values_jump)
            assert state in expected_q_values_per_state, \
                "ERROR: state '{}' not expected in q-table as it's a terminal state!".format(state)
            recursive_assert_almost_equal(q_values_forward, expected_q_values_per_state[state]["forward"], decimals=0)
            recursive_assert_almost_equal(q_values_jump, expected_q_values_per_state[state]["jump"], decimals=0)

    def test_double_dqn_on_4x4_grid_world(self):
        """
        Creates a double DQNAgent and runs it via a Runner on a simple 2x2 GridWorld.
        """
        dummy_env = GridWorld("4x4")
        agent_config = config_from_path("configs/dqn_agent_for_4x4_gridworld.json")
        preprocessing_spec = agent_config.pop("preprocessing_spec")
        agent = DQNAgent.from_spec(
            agent_config,
            dueling_q=False,
            state_space=self.grid_world_4x4_flattened_state_space,
            action_space=dummy_env.action_space,
            update_spec=dict(update_interval=4, batch_size=32, sync_interval=32),
            optimizer_spec=dict(type="adam", learning_rate=["linear", 0.01, 0.0001])
        )

        time_steps = 9000
        worker = SingleThreadedWorker(
            env_spec=lambda: GridWorld("4x4"),
            agent=agent,
            preprocessing_spec=preprocessing_spec,
            worker_executes_preprocessing=True
        )
        results = worker.execute_timesteps(time_steps, use_exploration=True)

        self.assertEqual(results["timesteps_executed"], time_steps)
        self.assertEqual(results["env_frames"], time_steps)
        self.assertGreaterEqual(results["mean_episode_reward"], -7.0)
        self.assertGreaterEqual(results["max_episode_reward"], 0.0)
        self.assertLessEqual(results["episodes_executed"], time_steps / 5)

        # Check all learnt Q-values.
        q_values = agent.graph_executor.execute(
            ("get_q_values", one_hot(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]), depth=16))
        )
        # Walking towards goal is always goo.
        self.assertTrue(q_values[0][0] < q_values[0][1])
        self.assertTrue(q_values[6][3] < q_values[6][1])
        # Falling into holes.
        self.assertTrue(q_values[1][1] < -4.0)  # Going right in state 1 is very bad
        self.assertTrue(q_values[4][2] < -4.0)  # Going down in state 4 is very bad
        self.assertTrue(q_values[12][2] < -4.0)  # Going down in state 12 is very bad
        # Reaching goal.
        self.assertTrue(q_values[11][1] > 0.0)

    def test_dqn_on_cart_pole(self):
        """
        Creates a DQNAgent and runs it via a Runner on the CartPole Env.
        """
        dummy_env = OpenAIGymEnv("CartPole-v0")
        agent = DQNAgent.from_spec(
            config_from_path("configs/dqn_agent_for_cartpole.json"),
            double_q=False,
            dueling_q=False,
            state_space=dummy_env.state_space,
            action_space=dummy_env.action_space,
            execution_spec=dict(seed=15),
            update_spec=dict(update_interval=4, batch_size=24, sync_interval=64),
            optimizer_spec=dict(type="adam", learning_rate=0.05)
        )

        time_steps = 3000
        worker = SingleThreadedWorker(
            env_spec=lambda: OpenAIGymEnv("CartPole-v0", seed=15),
            agent=agent,
            render=self.is_windows,
            worker_executes_preprocessing=False
        )
        results = worker.execute_timesteps(time_steps, use_exploration=True)

        self.assertEqual(results["timesteps_executed"], time_steps)
        self.assertEqual(results["env_frames"], time_steps)
        self.assertGreaterEqual(results["mean_episode_reward"], 25)
        self.assertGreaterEqual(results["max_episode_reward"], 100.0)
        self.assertLessEqual(results["episodes_executed"], 200)

    def test_double_dueling_dqn_on_cart_pole(self):
        """
        Creates a double and dueling DQNAgent and runs it via a Runner on the CartPole Env.
        """
        gym_env = "CartPole-v0"
        dummy_env = OpenAIGymEnv(gym_env)
        config_ = config_from_path("configs/dqn_agent_for_cartpole.json")
        # Add dueling config to agent.
        config_["policy_spec"] = {"units_state_value_stream": 3, "action_adapter_spec": {"pre_network_spec": [
            {"type": "dense", "units": 3}
        ]}}
        agent = DQNAgent.from_spec(
            config_,
            double_q=True,
            dueling_q=True,
            state_space=dummy_env.state_space,
            action_space=dummy_env.action_space,
            execution_spec=dict(seed=13),
            update_spec=dict(update_interval=4, batch_size=64, sync_interval=16),
            optimizer_spec=dict(type="adam", learning_rate=["linear", 0.01, 0.00001])
        )

        time_steps = 6000
        worker = SingleThreadedWorker(
            env_spec=lambda: OpenAIGymEnv(gym_env, seed=10),
            agent=agent,
            render=self.is_windows,
            worker_executes_preprocessing=False
        )
        results = worker.execute_timesteps(time_steps, use_exploration=True)

        self.assertEqual(results["timesteps_executed"], time_steps)
        self.assertEqual(results["env_frames"], time_steps)
        self.assertGreaterEqual(results["mean_episode_reward"], 40)
        self.assertLessEqual(results["episodes_executed"], time_steps / 10)

    def test_double_dqn_on_2x2_grid_world_single_action_to_container(self):
        """
        Tests how dqn solves a mapping of a single integer to multiple actions (as opposed to using container
        actions).
        """
        # ftj = forward + turn + jump
        env_spec = dict(world="2x2", action_type="ftj", state_representation="xy+orientation")
        agent_config = config_from_path("configs/dqn_agent_for_2x2_gridworld_single_to_container.json")
        preprocessing_spec = agent_config.pop("preprocessing_spec")

        action_space = IntBox(0, 18)
        agent = DQNAgent.from_spec(
            agent_config,
            huber_loss=True,
            double_q=True,
            dueling_q=True,
            state_space=FloatBox(shape=(4,)),
            action_space=action_space
        )

        time_steps = 10000
        worker = SingleThreadedWorker(
            env_spec=lambda: GridWorld.from_spec(env_spec),
            agent=agent,
            preprocessing_spec=preprocessing_spec,
            worker_executes_preprocessing=True,
            render=False
        )
        results = worker.execute_timesteps(time_steps, use_exploration=True)
        print(results)
