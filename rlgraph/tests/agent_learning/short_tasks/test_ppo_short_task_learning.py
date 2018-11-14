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

import logging
import unittest

from rlgraph.environments import OpenAIGymEnv, GridWorld
from rlgraph.agents import PPOAgent
from rlgraph.execution import SingleThreadedWorker
from rlgraph.spaces import FloatBox
from rlgraph.utils import root_logger
from rlgraph.tests.test_util import config_from_path, recursive_assert_almost_equal


class TestPPOShortTaskLearning(unittest.TestCase):
    """
    Tests whether the PPO agent can learn in simple environments.
    """
    root_logger.setLevel(level=logging.INFO)

    grid_world_2x2_preprocessing_spec = [dict(
        type="reshape",
        flatten=True,
        flatten_categories=4
    )]
    # Preprocessed state spaces.
    grid_world_2x2_flattened_state_space = FloatBox(shape=(4,), add_batch_rank=True)
    grid_world_4x4_flattened_state_space = FloatBox(shape=(16,), add_batch_rank=True)

    def test_ppo_on_2x2_grid_world(self):
        """
        Creates a PPO Agent and runs it via a Runner on the 2x2 Grid World Env.
        """
        env = GridWorld(world="2x2")
        agent = PPOAgent.from_spec(
            config_from_path("configs/ppo_agent_for_2x2_gridworld.json"),
            state_space=self.grid_world_2x2_flattened_state_space,
            action_space=env.action_space,
        )

        time_steps = 300
        worker = SingleThreadedWorker(
            env_spec=lambda: env,
            agent=agent,
            worker_executes_preprocessing=True,
            preprocessing_spec=self.grid_world_2x2_preprocessing_spec
        )
        results = worker.execute_timesteps(time_steps, use_exploration=True)

        print(results)

        # Assume we have learned something.
        self.assertGreater(results["mean_return"], -0.1)

        # Check the last action probs for the 2 valid next_states (start (after a reset) and one below start).
        action_probs = results[3]["action_probs"].reshape((80, 4))
        next_states = results[3]["states"][:, 1:].reshape((80,))
        for s_, probs in zip(next_states, action_probs):
            # Start state:
            # - Assume we picked "right" in state=1 (in order to step into goal state).
            # - OR we picked "up" or "left" in state=0 (unlikely, but possible).
            if s_ == 0:
                recursive_assert_almost_equal(probs[0], 0.0, decimals=2)
                self.assertTrue(probs[1] > 0.99 or probs[2] > 0.99)
                recursive_assert_almost_equal(probs[3], 0.0, decimals=2)
            # One below start:
            # - Assume we picked "down" in start state with very large probability.
            # - OR we picked "left" or "down" in state=1 (unlikely, but possible).
            elif s_ == 1:
                recursive_assert_almost_equal(probs[0], 0.0, decimals=2)
                self.assertTrue(probs[1] > 0.99 or probs[2] > 0.99)
                recursive_assert_almost_equal(probs[3], 0.0, decimals=2)

        agent.terminate()

    def test_ppo_on_cart_pole(self):
        """
        Creates a PPO Agent and runs it via a Runner on the CartPole Env.
        """
        env = OpenAIGymEnv("CartPole-v0", seed=36)
        agent = PPOAgent.from_spec(
            config_from_path("configs/ppo_agent_for_cartpole.json"),
            state_space=env.state_space,
            action_space=env.action_space,
        )

        time_steps = 1000
        worker = SingleThreadedWorker(
            env_spec=lambda: env,
            agent=agent,
            worker_executes_preprocessing=False,
            render=True
        )
        results = worker.execute_timesteps(time_steps, use_exploration=True)

        print(results)

        self.assertEqual(results["timesteps_executed"], time_steps)
        self.assertEqual(results["env_frames"], time_steps)
        self.assertGreaterEqual(results["mean_episode_reward"], 25)
        self.assertGreaterEqual(results["max_episode_reward"], 100.0)
        self.assertLessEqual(results["episodes_executed"], 100)

