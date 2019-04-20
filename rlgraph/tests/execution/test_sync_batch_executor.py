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

import unittest
from rlgraph.execution.ray.sync_batch_executor import SyncBatchExecutor
from rlgraph.tests.test_util import config_from_path


class TestSyncBatchExecutor(unittest.TestCase):
    """
    Tests the synchronous batch executor which provides an interface for executing A2C-style algorithms.
    via Ray.
    """
    def test_ppo_learning_cartpole(self):
        """
        Tests if sync-batch ppo can solve cartpole.
        """
        env_spec = dict(
            type="openai",
            gym_env="CartPole-v0"
        )
        agent_config = config_from_path("configs/sync_batch_ppo_cartpole.json")

        executor = SyncBatchExecutor(
            environment_spec=env_spec,
            agent_config=agent_config,
        )
        # Define executor, test assembly.
        print("Successfully created executor.")

        # Executes actual workload.
        result = executor.execute_workload(workload=dict(num_timesteps=20000, report_interval=1000,
                                                         report_interval_min_seconds=1))
        print("Finished executing workload:")
        print(result)

    def test_learning_2x2_grid_world_container_actions(self):
        """
        Tests sync batch container action functionality.
        """
        env_spec = dict(
            type="grid-world",
            world="2x2",
            save_mode=False,
            action_type="ftj",
            state_representation="xy+orientation"
        )
        agent_config = config_from_path("configs/sync_batch_ppo_gridworld_with_container_actions.json")
        executor = SyncBatchExecutor(
            environment_spec=env_spec,
            agent_config=agent_config,
        )
        # Define executor, test assembly.
        print("Successfully created executor.")

        # Executes actual workload.
        result = executor.execute_workload(workload=dict(
            num_timesteps=10000, report_interval=100, report_interval_min_seconds=1)
        )
        print(result)

    def test_ppo_learning_pendulum(self):
        """
        Tests if sync-batch ppo can solve Pendulum.
        """
        env_spec = dict(
            type="openai",
            gym_env="Pendulum-v0"
        )
        agent_config = config_from_path("configs/sync_batch_ppo_pendulum.json")

        executor = SyncBatchExecutor(
            environment_spec=env_spec,
            agent_config=agent_config,
        )
        # Define executor, test assembly.
        print("Successfully created executor.")

        # Executes actual workload.
        result = executor.execute_workload(workload=dict(num_timesteps=500000, report_interval=25000,
                                                         report_interval_min_seconds=1))
        print("Finished executing workload:")
        print(result)
