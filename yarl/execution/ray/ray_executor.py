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

from six.moves import xrange
import logging
import numpy as np

from yarl import get_distributed_backend
from yarl.agents import Agent
from yarl.envs import Environment

if get_distributed_backend() == "ray":
    import ray


class RayExecutor(object):
    """
    Abstract distributed Ray executor.

    A Ray executor implements a specific distributed learning semantic by delegating
    distributed state management and execution to the Ray execution engine.

    """
    def __init__(self, cluster_spec):
        """

        Args:
            cluster_spec (dict): Contains all information necessary to set up and execute
                agents on a Ray cluster.
        """
        self.logger = logging.getLogger(__name__)

        # Ray workers for remote data collection.
        self.ray_remote_workers = None
        self.cluster_spec = cluster_spec

    def ray_init(self):
        """
        Connects to a Ray cluster or starts one if none exists.
        """
        self.logger.info("Initializing Ray cluster with cluster spec: {}".format(
            self.cluster_spec
        ))
        # Avoiding accidentally starting local redis clusters.
        if 'redis_host' not in self.cluster_spec:
            self.logger.warning("Warning: No redis host provided, starting local redis server.")
        ray.init(
            redis_address=self.cluster_spec['redis_host'] if 'redis_host' in self.cluster_spec else None,
            num_cpus=self.cluster_spec['ray_num_cpus'] if 'ray_num_cpus' in self.cluster_spec else None,
            num_gpus=self.cluster_spec['ray_num_gpus'] if 'ray_num_gpus' in self.cluster_spec else None,
        )

    def create_remote_workers(self, cls, num_actors, *args):
        """
        Creates Ray actors for remote execution.
        Args:
            cls (RayWorker): Actor class, must be an instance of RayWorker.
            num_actors (int): Num
            *args (any): Arguments for RayWorker class.
        Returns:
            list: Remote Ray actors.
        """
        return [cls.remote(args) for _ in xrange(num_actors)]

    def setup_execution(self):
        """
        Creates and initializes all remote agents on the Ray cluster.
        """
        raise NotImplementedError

    def execute_workload(self, workload):
        """
        Executes a given workload according to a specific distributed update semantic.

        Args:
            workload (dict): Dict specifying workload by describing environments, number of steps
                or episodes to execute, and termination conditions.

        Returns:
            dict: Summary statistics of distributed workload.
        """
        raise NotImplementedError

    @staticmethod
    def build_agent_from_config(agent_config):
        """
        Builds agent without using from_spec as Ray cannot handle kwargs correctly
        at the moment.

        Args:
            agent_config (dict): Agent config. Must contain 'type' field to lookup constructor.

        Returns:
            Agent: YARL agent object.
        """
        agent_cls = Agent.__lookup_classes__.get(agent_config.pop('type'))
        return agent_cls(**agent_config)

    @staticmethod
    def build_env_from_config(env_spec):
        """
        Builds environment without using from_spec as Ray cannot handle kwargs correctly
        at the moment.

        Args:
            env_spec (dict): Environment specificaton. Must contain 'type' field to lookup constructor.

        Returns:
            Environment: Env object.
        """
        env_cls = Environment.__lookup_classes__.get(env_spec['type'])
        return env_cls(env_spec['gym_env'])

    def get_worker_results(self):
        """
        Fetches execution statistics from remote workers and aggregates them.

        Returns:
            dict: Aggregate worker statistics.
        """
        min_rewards = []
        max_rewards = []
        mean_rewards = []
        final_rewards = []
        worker_op_throughputs = []
        worker_env_frame_throughputs = []
        episodes_executed = 0
        steps_executed = 0

        for ray_worker in self.ray_remote_workers:
            task = ray_worker.get_workload_statistics.remote()
            metrics = ray.get(task)
            min_rewards.append(metrics['min_episode_reward'])
            max_rewards.append(metrics['max_episode_reward'])
            mean_rewards.append(metrics['mean_episode_reward'])
            episodes_executed += metrics['episodes_executed']
            steps_executed += metrics['worker_steps']
            final_rewards.append(metrics['final_episode_reward'])
            worker_op_throughputs.append(metrics['mean_worker_ops_per_second'])
            worker_env_frame_throughputs.append(metrics['mean_worker_env_frames_per_second'])

        return dict(
            min_reward=np.min(min_rewards),
            max_reward=np.max(max_rewards),
            mean_reward=np.mean(mean_rewards),
            mean_final_reward=np.mean(final_rewards),
            episodes_executed=episodes_executed,
            steps_executed=steps_executed,
            # Identify potential straggling workers.
            mean_worker_op_throughput=np.mean(worker_op_throughputs),
            min_worker_op_throughput=np.min(worker_op_throughputs),
            max_worker_op_throughput=np.max(worker_op_throughputs),
            mean_worker_env_frame_throughput=np.mean(worker_op_throughputs)
        )