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

from yarl.agents import DQNAgent
from yarl.envs import GridWorld
from yarl.execution.single_threaded_worker import SingleThreadedWorker
from yarl.tests import root_logger


class TestDQNAgent(unittest.TestCase):
    """
    Tests the DQN Agent on simple deterministic learning tasks.
    """
    root_logger.setLevel(level=logging.INFO)

    def test_dqn_assembly(self):
        """
        Creates a DQNAgent and runs it for a few steps in the random Env.
        """
        env = RandomEnv(deterministic=True)
        agent = DQNAgent.from_spec("configs/test_simple_dqn_agent.json",
                                   state_space=env.state_space.add_rank(),
                                   action_space=env.action_space
                                   )

        worker = SingleThreadedWorker(agent=agent, env=env)
        results = worker.execute_timesteps(1000, deterministic=True)

        print(results)

        # TODO: assert good results :)

    def test_dqn_on_2x2_grid_world(self):
        """
        Creates a DQNAgent and runs it via a Runner on a simple 2x2 GridWorld.
        """
        env = GridWorld("2x2")
        agent = DQNAgent.from_spec("configs/test_simple_dqn_agent.json",
                                   state_space=env.state_space,
                                   action_space=env.action_space,

                                   )

        worker = SingleThreadedWorker(agent=agent, env=env)
        results = worker.execute_timesteps(1000, deterministic=True)

        print(results)

        # TODO: assert good results :)
