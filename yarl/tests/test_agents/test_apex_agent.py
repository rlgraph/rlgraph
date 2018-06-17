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
from yarl.envs import RandomEnv
from yarl.execution.single_threaded_worker import SingleThreadedWorker
from yarl.utils import root_logger


class TestApexAgent(unittest.TestCase):
    """
    Tests the Apex Agent on simple deterministic learning tasks.
    """
    root_logger.setLevel(level=logging.INFO)

    def test_apex_assembly(self):
        """
        Creates a DQNAgent and runs it for a few steps in the random Env.
        """
        env = RandomEnv(state_space=spaces.IntBox(2), action_space=spaces.IntBox(2), deterministic=True)
        agent = ApexAgent.from_spec(
            "configs/test_dqn_agent_for_random_env.json",
            state_space=env.state_space,
            action_space=env.action_space
        )

        worker = SingleThreadedWorker(environment=env, agent=agent)
        results = worker.execute_timesteps(1000, deterministic=True)

        self.assertEqual(results["timesteps_executed"], 1000)
        self.assertEqual(results["env_frames"], 1000)
