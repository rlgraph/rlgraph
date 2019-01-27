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

from rlgraph.environments import OpenAIGymEnv, GridWorld
from rlgraph.agents import PPOAgent
from rlgraph.execution import SingleThreadedWorker
from rlgraph.utils import root_logger
from rlgraph.tests.test_util import config_from_path


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

        # Assume we have learned something.
        self.assertGreater(results["mean_episode_reward"], -0.2)

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

        time_steps = 3000
        worker = SingleThreadedWorker(
            env_spec=lambda: env,
            agent=agent,
            worker_executes_preprocessing=False,
            render=self.is_windows
        )
        results = worker.execute_timesteps(time_steps, use_exploration=True)

        print(results)

        self.assertEqual(results["timesteps_executed"], time_steps)
        self.assertEqual(results["env_frames"], time_steps)
        self.assertLessEqual(results["episodes_executed"], time_steps / 10)

    def test_ppo_on_pendulum(self):
        """
        Creates a PPO Agent and runs it via a Runner on the Pendulum env.
        """
        env = OpenAIGymEnv("Pendulum-v0")
        agent = PPOAgent.from_spec(
            config_from_path("configs/ppo_agent_for_pendulum.json"),
            state_space=env.state_space,
            action_space=env.action_space
        )

        worker = SingleThreadedWorker(
            env_spec=lambda: env,
            agent=agent,
            worker_executes_preprocessing=False,
            render=self.is_windows
        )
        results = worker.execute_episodes(500, use_exploration=True)

        print(results)


