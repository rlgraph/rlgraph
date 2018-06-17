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

import unittest
from time import sleep

from yarl import get_distributed_backend
from yarl.execution.ray import ApexExecutor

if get_distributed_backend() == "ray":
    import ray


class TestRayExecutor(unittest.TestCase):
    """
    Tests the ApexExecutor which provides an interface for distributing Apex-style workloads
    via Ray.
    """

    env_spec = dict(
      type="openai",
      gym_env="CartPole-v0"
    )
    agent_config = dict(
        type="random"
    )

    # Note that all of these args are allowed to be none,
    # Ray will just start a local redis and fetch number of cpus.
    cluster_spec = dict(
        redis_address=None,
        num_cpus=2,
        num_gpus=0,
        task_queue_depth=1,
        weight_sync_steps=100,
        env_interaction_task_depth=1,
        num_worker_samples=100
    )

    def test_apex_workload(self):
        # Define executor, test assembly.
        executor = ApexExecutor(
            environment_spec=self.env_spec,
            agent_config=self.agent_config,
            cluster_spec=self.cluster_spec
        )
        print("Successfully created executor.")

        # Kicks off remote tasks.
        executor.init_tasks()
        print("Initialized Apex executor Ray tasks, starting workload:")

        # Executes actual workload.
        result = executor.execute_workload(workload=dict(num_timesteps=10000))
        print("Finished executing workload:")
        print(result)
