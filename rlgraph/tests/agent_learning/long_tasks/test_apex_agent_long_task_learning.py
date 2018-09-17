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

import ray
import unittest
import time
from rlgraph.agents import ApexAgent
from rlgraph.environments import OpenAIGymEnv
from rlgraph.execution.ray.apex import ApexExecutor
from rlgraph.execution.ray import RayWorker
from rlgraph.tests.test_util import config_from_path


class TestApexAgentLongTaskLearning(unittest.TestCase):
    """
    Tests whether the Apex Agent can start learning in pong.

    WARNING: This test requires large amounts of memory due to large buffer size.
    """
    env_spec = dict(
        type="openai",
        gym_env="PongNoFrameskip-v4",
        # The frameskip in the agent config will trigger worker skips, this
        # is used for internal env.
        frameskip=4,
        max_num_noops=30,
        episodic_life=False,
        fire_reset=True
    )

    def test_worker_init(self):
        """
        Tests if workers initialize without problems for the pong config.
        """
        agent_config = config_from_path("configs/ray_apex_for_pong.json")

        # Long initialization times can lead to Ray crashes.
        start = time.monotonic()
        executor = ApexExecutor(
            environment_spec=self.env_spec,
            agent_config=agent_config,
        )
        end = time.monotonic() - start
        print("Initialized {} workers in {} s.".format(
            executor.num_sample_workers, end
        ))
        executor.test_worker_init()

    def test_worker_update(self):
        """
        Tests if a worker can update from an external batch correct including all
        corrections and postprocessing using the pong spec.

        N.b. this test does not use Ray.
        """
        ray.init()
        agent_config = config_from_path("configs/ray_apex_for_pong.json")
        ray_spec = agent_config["execution_spec"].pop("ray_spec")
        worker_cls = RayWorker.as_remote().remote
        ray_spec["worker_spec"]["worker_sample_size"] = 198
        ray_spec["worker_spec"]["worker_executes_exploration"] = True
        ray_spec["worker_spec"]["ray_exploration"] = 0.4

        worker = worker_cls(agent_config, ray_spec["worker_spec"], self.env_spec,)
        build_result = worker.init_agent.remote()
        ready, not_ready = ray.wait([build_result], num_returns=1)
        result = ray.get(ready)
        print(result)
        time.sleep(5)

        start = time.perf_counter()
        task = worker.execute_and_get_with_count.remote()
        result, count = ray.get(task)
        task_time = time.perf_counter() - start
        print("internal result metrics = {}, external task time = {},"
              "external throughput = {}".format(result.get_metrics(), task_time, 198 / task_time))

    def test_initial_training_pong(self):
        """
        Tests if Apex can start learning pong effectively on ray.
        """
        agent_config = config_from_path("configs/ray_apex_for_pong.json")
        executor = ApexExecutor(
            environment_spec=self.env_spec,
            agent_config=agent_config,
        )

        # Executes actual workload.
        result = executor.execute_workload(workload=dict(
            num_timesteps=10000000, report_interval=10000, report_interval_min_seconds=10)
        )
        print("Finished executing workload:")
        print(result)
