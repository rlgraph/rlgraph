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

import unittest

from rlgraph.agents.random_agent import RandomAgent
from rlgraph.environments import OpenAIGymEnv
from rlgraph.execution.single_threaded_worker import SingleThreadedWorker


class TestSingleThreadedWorker(unittest.TestCase):

    environment = OpenAIGymEnv(gym_env='CartPole-v0')

    def test_timesteps(self):
        """
        Simply tests if timestep execution loop works and returns a result.
        """
        agent = RandomAgent(
            action_space=self.environment.action_space,
            state_space=self.environment.state_space
        )
        worker = SingleThreadedWorker(
            env_spec=lambda: self.environment,
            agent=agent,
            frameskip=1,
            worker_executes_preprocessing=False
        )

        result = worker.execute_timesteps(100)
        self.assertEqual(result['timesteps_executed'], 100)
        self.assertGreater(result['episodes_executed'], 0)
        self.assertLessEqual(result['episodes_executed'], 100)
        self.assertGreaterEqual(result['env_frames'], 100)
        self.assertGreaterEqual(result['runtime'], 0.0)

    def test_episodes(self):
        """
        Simply tests if episode execution loop works and returns a result.
        """
        agent = RandomAgent(
            action_space=self.environment.action_space,
            state_space=self.environment.state_space
        )
        worker = SingleThreadedWorker(
            env_spec=lambda: self.environment,
            agent=agent,
            frameskip=1,
            worker_executes_preprocessing=False
        )

        result = worker.execute_episodes(5, max_timesteps_per_episode=10)
        # Max 5 * 10.
        self.assertLessEqual(result['timesteps_executed'], 50)
        self.assertEqual(result['episodes_executed'], 5)
        self.assertLessEqual(result['env_frames'], 50)
        self.assertGreaterEqual(result['runtime'], 0.0)
