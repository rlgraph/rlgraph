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

from rlgraph.execution.ray.apex import ApexExecutor
from rlgraph.tests.test_util import config_from_path


class TestApexExecutor(unittest.TestCase):
    """
    Tests the ApexExecutor which provides an interface for distributing Apex-style workloads
    via Ray.
    """
    def test_learning_grid_world(self):
        """
        Tests if apex can learn a simple environment using a single worker, thus replicating
        dqn.
        """
        env_spec = dict(
            type="gridworld",
            world="2x2",
            save_mode=False
        )
        agent_config = config_from_path("configs/apex_agent_gridworld_for_2x2_grid.json")

        executor = ApexExecutor(
            environment_spec=env_spec,
            agent_config=agent_config,
        )
        # Define executor, test assembly.
        print("Successfully created executor.")

        # Executes actual workload.
        result = executor.execute_workload(workload=dict(num_timesteps=1000, report_interval=100,
                                                         report_interval_min_seconds=1))
        full_worker_stats = executor.result_by_worker()
        print("All finished episode rewards")
        print(full_worker_stats["episode_rewards"])

    def test_learning_cartpole(self):
        """
        Tests if apex can learn a simple environment using a single worker, thus replicating
        dqn.
        """
        env_spec = dict(
            type="openai",
            gym_env="CartPole-v0"
        )
        agent_config = config_from_path("configs/apex_agent_cartpole.json")
        executor = ApexExecutor(
            environment_spec=env_spec,
            agent_config=agent_config,
        )
        # Define executor, test assembly.
        print("Successfully created executor.")

        # Executes actual workload.
        result = executor.execute_workload(workload=dict(num_timesteps=10000, report_interval=1000,
                                                         report_interval_min_seconds=1))
        print("Finished executing workload:")
        print(result)

        full_worker_stats = executor.result_by_worker()
        print("All finished episode rewards")
        print(full_worker_stats["episode_rewards"])
