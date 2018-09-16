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
import time

from rlgraph import get_backend
from rlgraph.agents import DQNAgent
from rlgraph.environments import OpenAIGymEnv
from rlgraph.execution import SingleThreadedWorker
from rlgraph.tests.test_util import config_from_path
from rlgraph.utils import root_logger


class TestPytorchBackend(unittest.TestCase):
    """
    Tests PyTorch component execution.

    # TODO: This is a temporary test. We will later run all backend-specific
    tests via setting the executor in the component-test.
    """
    root_logger.setLevel(level=logging.INFO)

    def test_with_worker(self):
        env = OpenAIGymEnv("CartPole-v0")
        agent_config = config_from_path("configs/backend_performance_dqn_cartpole.json")

        # Test cpu settings for batching here.
        if get_backend() == "pytorch":
            agent_config["memory_spec"]["type"] = "mem_prioritized_replay"
        agent_config["update_spec"] = None

        agent = DQNAgent.from_spec(
            # Uses 2015 DQN parameters as closely as possible.
            agent_config,
            state_space=env.state_space,
            # Try with "reduced" action space (actually only 3 actions, up, down, no-op)
            action_space=env.action_space
        )

        worker = SingleThreadedWorker(
            env_spec=lambda: OpenAIGymEnv("CartPole-v0"),
            agent=agent,
            frameskip=1,
            num_envs=1
        )

        result = worker.execute_timesteps(1000)
        print(result)
