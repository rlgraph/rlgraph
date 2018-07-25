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

import json
import os
import unittest

from rlgraph.agents import ApexAgent
from rlgraph.environments import OpenAIGymEnv
from rlgraph.execution.ray import ApexExecutor


class TestApexAgentLongTaskLearning(unittest.TestCase):
    """
    Tests whether the Apex Agent can start learning in pong.

    WARNING: This test requires large amounts of memory due to large buffer size.
    """
    env_spec = dict(
      type="openai",
          gym_env="PongNoFrameskip-v4"
    )

    def test_agent_compilation(self):
        """
        Tests agent compilation without Ray to ease debugging on Windows.
        """
        path = os.path.join(os.getcwd(), "../configs/ray_apex_for_pong.json")
        with open(path, 'rt') as fp:
            agent_config = json.load(fp)
            # Remove.
            agent_config["execution_spec"].pop("ray_spec")
        environment = OpenAIGymEnv("PongNoFrameskip-v4", frameskip=4)

        agent_config["state_space"] = environment.state_space
        agent_config["action_space"] = environment.action_space
        agent = ApexAgent.from_spec(agent_config)
        print('Compiled apex agent')

    def test_initial_training_pong(self):
        """
        Tests if Apex can start learning pong effectively on ray.
        """
        path = os.path.join(os.getcwd(), "../configs/ray_apex_for_pong.json")
        with open(path, 'rt') as fp:
            agent_config = json.load(fp)

        executor = ApexExecutor(
            environment_spec=self.env_spec,
            agent_config=agent_config,
        )

        # Executes actual workload.
        result = executor.execute_workload(workload=dict(num_timesteps=50000000, report_interval=50000))
        print("Finished executing workload:")
        print(result)
