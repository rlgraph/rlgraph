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

from rlgraph.execution.ray.ray_policy_worker import RayPolicyWorker

from rlgraph import get_distributed_backend
from rlgraph.execution.ray.ray_executor import RayExecutor
from rlgraph.execution.ray.ray_util import merge_samples, RayWeight

if get_distributed_backend() == "ray":
    import ray


class SyncBatchExecutor(RayExecutor):
    """
    Implements distributed synchronous execution.
    """
    def __init__(self, environment_spec, agent_config):
        """
        Args:
            environment_spec (dict): Environment spec. Each worker in the cluster will instantiate
                an environment using this spec.
            agent_config (dict): Config dict containing agent and execution specs.
        """
        ray_spec = agent_config["execution_spec"].pop("ray_spec")
        self.worker_spec = ray_spec.pop("worker_spec")
        self.compress_states = self.worker_spec["compress_states"]
        super(SyncBatchExecutor, self).__init__(executor_spec=ray_spec.pop("executor_spec"),
                                           environment_spec=environment_spec,
                                           worker_spec=self.worker_spec)

        # Must specify an agent type.
        assert "type" in agent_config
        self.agent_config = agent_config
        environment = RayExecutor.build_env_from_config(self.environment_spec)
        self.agent_config["state_space"] = environment.state_space
        self.agent_config["action_space"] = environment.action_space

        self.local_agent = self.build_agent_from_config(self.agent_config)
        self.update_batch_size = self.agent_config["update_spec"]["batch_size"]

        # Create remote sample workers based on ray cluster spec.
        self.num_sample_workers = self.executor_spec["num_sample_workers"]

        # These are the tasks actually interacting with the environment.
        self.worker_sample_size = self.executor_spec["num_worker_samples"]

        assert not ray_spec, "ERROR: ray_spec still contains items: {}".format(ray_spec)
        self.logger.info("Setting up execution for Apex executor.")
        self.setup_execution()

    def setup_execution(self):
        # Start Ray cluster and connect to it.
        self.ray_init()

        # Create local worker agent according to spec.
        # Extract states and actions space.
        environment = RayExecutor.build_env_from_config(self.environment_spec)
        self.agent_config["state_space"] = environment.state_space
        self.agent_config["action_space"] = environment.action_space

        # Create remote workers for data collection.
        self.worker_spec["worker_sample_size"] = self.worker_sample_size
        self.logger.info("Initializing {} remote data collection agents, sample size: {}".format(
            self.num_sample_workers, self.worker_spec["worker_sample_size"]))
        self.ray_env_sample_workers = self.create_remote_workers(
            RayPolicyWorker, self.num_sample_workers, self.agent_config,
            # *args
            self.worker_spec, self.environment_spec, self.worker_frameskip
        )

    def _execute_step(self):
        """
        Executes a workload on Ray. The main loop performs the following
        steps until the specified number of steps or episodes is finished:

        - Sync weights to policy workers.
        - Schedule a set of samples
        - Wait until enough samples tasks are complete to form an update batch
        - Merge samples
        - Perform local update(s)
        """
        # Env steps done during this rollout.
        env_steps = 0

        # 1. Sync local learners weights to remote workers.
        weights = ray.put(RayWeight(self.local_agent.get_weights()))
        for ray_worker in self.ray_env_sample_workers:
            ray_worker.set_weights.remote(weights)

        # 2. Schedule samples and fetch results from RayWorkers.
        sample_batches = []
        num_samples = 0
        while num_samples < self.update_batch_size:
            batches = ray.get([worker.execute_and_get_timesteps.remote(self.worker_sample_size)
                              for worker in self.ray_env_sample_workers])
            # Each batch has exactly worker_sample_size length.
            num_samples += len(batches) * self.worker_sample_size
            sample_batches.extend(batches)

        env_steps += num_samples
        # 3. Merge samples
        batch = merge_samples(sample_batches, decompress=self.compress_states)

        # 4. Update from merged batch.
        self.local_agent.update(batch, apply_postprocessing=False)
        return env_steps, 1, 0, 0


