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

from rlgraph.agents import PPOAgent
from rlgraph.environments import OpenAIGymEnv, GridWorld
from rlgraph.execution import SingleThreadedWorker
from rlgraph.spaces import FloatBox
from rlgraph.tests.test_util import config_from_path, recursive_assert_almost_equal
from rlgraph.utils import root_logger
from rlgraph.utils.numpy import one_hot


class TestPPOShortTaskLearning(unittest.TestCase):
    """
    Tests whether the PPO agent can learn in simple environments.
    """
    root_logger.setLevel(level=logging.INFO)

    is_windows = os.name == "nt"

    def test_ppo_on_2x2_grid_world(self):
        """
        Creates a PPO Agent and runs it via a Runner on the 2x2 Grid World env.
        """
        env = GridWorld(world="2x2")
        agent = PPOAgent.from_spec(
            config_from_path("configs/ppo_agent_for_2x2_gridworld.json"),
            state_space=GridWorld.grid_world_2x2_flattened_state_space,
            action_space=env.action_space,
            execution_spec=dict(seed=15),
        )

        time_steps = 3000
        worker = SingleThreadedWorker(
            env_spec=lambda: env,
            agent=agent,
            worker_executes_preprocessing=True,
            preprocessing_spec=GridWorld.grid_world_2x2_preprocessing_spec
        )
        results = worker.execute_timesteps(time_steps, use_exploration=True)

        print(results)

        # Check value function outputs for states 0 and 1.
        # NOTE that this test only works if standardize-advantages=False.
        values = agent.graph_executor.execute(
            ("get_state_values", np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]))
        )[:, 0]
        recursive_assert_almost_equal(values[0], 0.0, decimals=1)  # state 0 should have a value of 0.0
        recursive_assert_almost_equal(values[1], 1.0, decimals=1)  # state 1 should have a value of +1.0

        self.assertEqual(results["timesteps_executed"], time_steps)
        self.assertEqual(results["env_frames"], time_steps)
        self.assertLessEqual(results["episodes_executed"], time_steps / 2)
        # Assume we have learned something.
        self.assertGreater(results["mean_episode_reward"], -0.3)

    def test_ppo_on_2x2_grid_world_with_container_actions(self):
        """
        Creates a PPO agent and runs it via a Runner on a simple 2x2 GridWorld using container actions.
        """
        # -----
        # |^|H|
        # -----
        # | |G|  ^=start, looking up
        # -----

        # ftj = forward + turn + jump
        env_spec = dict(world="2x2", action_type="ftj", state_representation="xy+orientation")
        dummy_env = GridWorld.from_spec(env_spec)
        agent_config = config_from_path("configs/ppo_agent_for_2x2_gridworld_with_container_actions.json")
        preprocessing_spec = agent_config.pop("preprocessing_spec")

        agent = PPOAgent.from_spec(
            agent_config,
            state_space=FloatBox(shape=(4,)),
            action_space=dummy_env.action_space
        )

        time_steps = 5000
        worker = SingleThreadedWorker(
            env_spec=lambda: GridWorld.from_spec(env_spec),
            agent=agent,
            preprocessing_spec=preprocessing_spec,
            worker_executes_preprocessing=True,
            render=False
        )
        results = worker.execute_timesteps(time_steps, use_exploration=True)

        print(results)

        self.assertEqual(results["timesteps_executed"], time_steps)
        self.assertEqual(results["env_frames"], time_steps)
        self.assertLessEqual(results["episodes_executed"], time_steps)
        # Assume we have learned something.
        self.assertGreaterEqual(results["mean_episode_reward"], -2.0)

    def test_ppo_on_4x4_grid_world(self):
        """
        Creates a PPO Agent and runs it via a Runner on the 4x4 Grid World env.
        """
        env = GridWorld(world="4x4")
        agent = PPOAgent.from_spec(
            config_from_path("configs/ppo_agent_for_4x4_gridworld.json"),
            state_space=GridWorld.grid_world_4x4_flattened_state_space,
            action_space=env.action_space
        )

        time_steps = 3000
        worker = SingleThreadedWorker(
            env_spec=lambda: env,
            agent=agent,
            worker_executes_preprocessing=True,
            preprocessing_spec=GridWorld.grid_world_4x4_preprocessing_spec,
            episode_finish_callback=lambda episode_return, duration, timesteps, **kwargs: print(
                "Episode done return={} timesteps={}".format(episode_return, timesteps)),
        )
        results = worker.execute_timesteps(time_steps, use_exploration=True)

        print(results)

        # Check value function outputs for states on a good trajectory.
        # NOTE that this test only works if standardize-advantages=False.
        values = agent.graph_executor.execute(("get_state_values", one_hot(np.array([1, 2, 6, 7, 11]), depth=16)))[:, 0]
        recursive_assert_almost_equal(values, np.array([0.6, 0.7, 0.8, 0.9, 1.0]), decimals=1)

        self.assertEqual(results["timesteps_executed"], time_steps)
        self.assertEqual(results["env_frames"], time_steps)
        self.assertLessEqual(results["episodes_executed"], time_steps / 4)
        # Assume we have learned something.
        self.assertGreater(results["mean_episode_reward"], -6.0)

    def test_ppo_on_4_room_grid_world(self):
        """
        Creates a PPO agent and runs it via a Runner on a 4-rooms GridWorld.
        """
        env_spec = dict(world="4-room")
        dummy_env = GridWorld.from_spec(env_spec)
        agent_config = config_from_path("configs/ppo_agent_for_4_room_gridworld.json")
        preprocessing_spec = agent_config.pop("preprocessing_spec")

        agent = PPOAgent.from_spec(
            agent_config,
            state_space=FloatBox(shape=(dummy_env.state_space.num_categories,)),
            action_space=dummy_env.action_space
        )

        episodes = 30
        worker = SingleThreadedWorker(
            env_spec=lambda: GridWorld.from_spec(env_spec),
            agent=agent,
            preprocessing_spec=preprocessing_spec,
            worker_executes_preprocessing=True,
            render=False,
            episode_finish_callback=lambda episode_return, duration, timesteps, **kwargs: print("Episode done return={}".format(episode_return)),
            #update_finish_callback=lambda loss: print("Update policy+vf-loss={}".format(loss[0]))
        )
        results = worker.execute_episodes(
            num_episodes=episodes, max_timesteps_per_episode=10000, use_exploration=True
        )

        print(results)

        # Check value function outputs for states 46 (doorway close to goal) and start state.
        values = agent.graph_executor.execute(
            ("get_state_values", np.array([one_hot(np.array(46), depth=121), one_hot(np.array(30), depth=121)]))
        )[:, 0]
        recursive_assert_almost_equal(values[0], -1.0, decimals=1)  # state 46 should have a value of -1.0
        recursive_assert_almost_equal(values[1], -50, decimals=1)  # state 30 (start) should have a value of ???

        self.assertEqual(results["episodes_executed"], episodes)
        # Assume we have learned something.
        #self.assertGreaterEqual(results["mean_episode_reward"], -2.0)

    def test_ppo_on_cart_pole(self):
        """
        Creates a PPO Agent and runs it via a Runner on the CartPole env.
        """
        env = OpenAIGymEnv("CartPole-v0", seed=36)
        agent = PPOAgent.from_spec(
            config_from_path("configs/ppo_agent_for_cartpole.json"),
            state_space=env.state_space,
            action_space=env.action_space
        )

        time_steps = 5000
        worker = SingleThreadedWorker(
            env_spec=lambda: env,
            agent=agent,
            worker_executes_preprocessing=False,
            render=False,  #self.is_windows
            episode_finish_callback=lambda episode_return, duration, timesteps, env_num:
            print("episode return {}; steps={}".format(episode_return, timesteps))
        )
        results = worker.execute_timesteps(time_steps, use_exploration=True)

        print(results)

        self.assertEqual(results["timesteps_executed"], time_steps)
        self.assertEqual(results["env_frames"], time_steps)
        self.assertLessEqual(results["episodes_executed"], time_steps / 10)
        # Assume we have learned something.
        self.assertGreaterEqual(results["mean_episode_reward"], 40.0)

    def test_ppo_on_pendulum(self):
        """
        Creates a PPO Agent and runs it via a Runner on the Pendulum env.
        """
        env_spec = dict(type="openai-gym", gym_env="Pendulum-v0")
        dummy_env = OpenAIGymEnv.from_spec(env_spec)
        agent_spec = config_from_path("configs/ppo_agent_for_pendulum.json")
        preprocessing_spec = agent_spec.pop("preprocessing_spec", None)
        agent = PPOAgent.from_spec(
            agent_spec,
            state_space=dummy_env.state_space,
            action_space=dummy_env.action_space
        )

        worker = SingleThreadedWorker(
            env_spec=env_spec,
            num_environments=10,
            agent=agent,
            preprocessing_spec=preprocessing_spec,
            worker_executes_preprocessing=True,
            render=False, #self.is_windows,
            episode_finish_callback=lambda episode_return, duration, timesteps, env_num:
            print("episode return {}; steps={}".format(episode_return, timesteps))
        )
        results = worker.execute_timesteps(num_timesteps=int(5e6), use_exploration=True)

        print(results)

    def test_ppo_on_lunar_lander(self):
        """
        Creates a PPO Agent and runs it via a Runner on the Pendulum env.
        """
        env = OpenAIGymEnv("LunarLander-v2")
        agent = PPOAgent.from_spec(
            config_from_path("configs/ppo_agent_for_pendulum.json"),
            state_space=env.state_space,
            action_space=env.action_space
        )

        worker = SingleThreadedWorker(
            env_spec=lambda: env,
            agent=agent,
            worker_executes_preprocessing=False,
            render=False,  #self.is_windows,
            episode_finish_callback=lambda episode_return, duration, timesteps, env_num:
            print("episode return {}; steps={}".format(episode_return, timesteps))
        )
        results = worker.execute_timesteps(num_timesteps=int(5e6), use_exploration=True)

        print(results)
