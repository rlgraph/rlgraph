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

import logging
import unittest
from yarl.agents import ApexAgent
import yarl.spaces as spaces
from yarl.environments import RandomEnv
from yarl.execution.single_threaded_worker import SingleThreadedWorker
from yarl.utils import root_logger


class TestApexAgent(unittest.TestCase):
    """
    Tests the ApexAgent assembly on the RandomEnv.
    """
    root_logger.setLevel(level=logging.INFO)

    def test_apex_assembly(self):
        """
        Creates an ApexAgent and runs it for a few steps in the RandomEnv.
        """
        env = RandomEnv(state_space=spaces.IntBox(2), action_space=spaces.IntBox(2), deterministic=True)
        agent = ApexAgent.from_spec(
            "configs/apex_agent_for_random_env.json",
            state_space=env.state_space,
            preprocessed_state_space=spaces.FloatBox(shape=(2,)),  # TODO: remove once auto preprocessor Space inference done.
            action_space=env.action_space
        )

        worker = SingleThreadedWorker(environment=env, agent=agent)
        timesteps = 100
        results = worker.execute_timesteps(timesteps, use_exploration=False)

        print(results)

        self.assertEqual(results["timesteps_executed"], timesteps)
        self.assertEqual(results["env_frames"], timesteps)
        # Assert deterministic execution of Env and Agent.
        self.assertAlmostEqual(results["mean_episode_reward"], 5.923551400230593)
        self.assertAlmostEqual(results["max_episode_reward"], 14.312868008192979)
        self.assertAlmostEqual(results["final_episode_reward"], 0.14325251090518198)

    def test_apex_weight_syncing(self):
        env = RandomEnv(state_space=spaces.IntBox(2), action_space=spaces.IntBox(2), deterministic=True)

        agent = ApexAgent.from_spec(
            "configs/apex_agent_for_random_env.json",
            state_space=env.state_space,
            action_space=env.action_space,
            preprocessed_state_space=env.state_space
        )

        policy_weights = agent.get_policy_weights()
        print('policy weights: {}'.format(policy_weights))

