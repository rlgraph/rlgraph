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

from rlgraph.environments import OpenAIGymEnv
from rlgraph.agents import ActorCriticAgent
from rlgraph.execution import SingleThreadedWorker
from rlgraph.utils import root_logger
from rlgraph.tests.test_util import config_from_path, recursive_assert_almost_equal


class TestActorCriticShortTaskLearning(unittest.TestCase):
    """
    Tests whether the Actor-critic can learn in simple environments.
    """
    root_logger.setLevel(level=logging.INFO)

    def test_actor_critic_on_cart_pole(self):
        """
        Creates an Actor-critic and runs it via a Runner on the CartPole Env.
        """
        env = OpenAIGymEnv("CartPole-v0")
        agent = ActorCriticAgent.from_spec(
            config_from_path("configs/actor_critic_agent_for_cartpole.json"),
            state_space=env.state_space,
            action_space=env.action_space,
        )

        time_steps = 10000
        worker = SingleThreadedWorker(
            env_spec=lambda: OpenAIGymEnv("CartPole-v0", seed=15),
            agent=agent,
            worker_executes_preprocessing=False
        )
        results = worker.execute_timesteps(time_steps, use_exploration=True)

        print(results)

        # self.assertEqual(results["timesteps_executed"], time_steps)
        # self.assertEqual(results["env_frames"], time_steps)
        # self.assertGreaterEqual(results["mean_episode_reward"], 25)
        # self.assertGreaterEqual(results["max_episode_reward"], 100.0)
        # self.assertLessEqual(results["episodes_executed"], 100)


