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
import numpy as np
import os
import unittest

from rlgraph.environments import GaussianDensityAsRewardEnvironment
from rlgraph.agents import SACAgent
from rlgraph.execution import SingleThreadedWorker
from rlgraph.utils import root_logger
from rlgraph.tests.test_util import config_from_path


class TestSACShortTaskLearning(unittest.TestCase):
    """
    Tests whether the PPO agent can learn in simple environments.
    """
    root_logger.setLevel(level=logging.INFO)

    is_windows = os.name == "nt"

    def test_sac_learning_on_gaussian_density_as_reward_env(self):
        """
        Creates an SAC-Agent and runs it via a Runner on the GaussianDensityAsRewardEnvironment.
        """
        env = GaussianDensityAsRewardEnvironment(episode_length=5)
        agent = SACAgent.from_spec(
            config_from_path("configs/sac_agent_for_functionality_test.json"),
            state_space=env.state_space,
            action_space=env.action_space
        )

        worker = SingleThreadedWorker(
            env_spec=lambda: env, agent=agent
        )
        worker.execute_episodes(num_episodes=500)
        rewards = worker.finished_episode_rewards[0]  # 0=1st env in vector-env
        assert np.mean(rewards[:100]) < np.mean(rewards[-100:])

        worker.execute_episodes(num_episodes=100, use_exploration=False, update_spec=None)
        rewards = worker.finished_episode_rewards[0]
        assert len(rewards) == 100
        evaluation_score = np.mean(rewards)
        assert .5 * env.get_max_reward() < evaluation_score <= env.get_max_reward()

