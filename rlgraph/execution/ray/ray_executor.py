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

from copy import deepcopy

from rlgraph.execution.ray.ray_util import worker_exploration
from six.moves import xrange as range_
import logging
import numpy as np
import time

from rlgraph import get_distributed_backend
from rlgraph.agents import Agent
from rlgraph.environments import Environment

if get_distributed_backend() == "ray":
    import ray


class RayExecutor(object):
    """
    Abstract distributed Ray executor.

    A Ray executor implements a specific distributed learning semantic by delegating
    distributed state management and execution to the Ray execution engine.
    """
    def __init__(self, executor_spec, environment_spec, worker_spec):
        """
        Args:
            executor_spec (dict): Contains all information necessary to set up and execute
                agents on a Ray cluster.
            environment_spec (dict): Environment spec. Each worker in the cluster will instantiate
                an environment using this spec.
            worker_spec (dict): Worker spec to read out for reporting.
        """
        self.logger = logging.getLogger(__name__)

        # Ray workers for remote data collection.
        self.ray_env_sample_workers = None
        self.executor_spec = executor_spec
        self.num_cpus_per_worker = executor_spec.get("num_cpus_per_worker", 1)
        self.num_gpus_per_worker = executor_spec.get("num_gpus_per_worker", 0)
        self.environment_spec = environment_spec

        # Global performance metrics.
        self.sample_iteration_throughputs = None
        self.update_iteration_throughputs = None
        self.iteration_times = None
        self.worker_frameskip = worker_spec.get("frame_skip", 1)
        self.env_internal_frame_skip = environment_spec.get("frameskip", 1)

        # Map worker objects to host ids.
        self.worker_ids = dict()

    def ray_init(self):
        """
        Connects to a Ray cluster or starts one if none exists.
        """
        self.logger.info("Initializing Ray cluster with executor spec:")
        for spec_key, value in self.executor_spec.items():
            self.logger.info("{}: {}".format(spec_key, value))

        # Avoiding accidentally starting local redis clusters.
        if 'redis_address' not in self.executor_spec:
            self.logger.warning("Warning: No redis address provided, starting local redis server.")
        ray.init(
            redis_address=self.executor_spec.get('redis_address', None),
            num_cpus=self.executor_spec.get('num_cpus', None),
            num_gpus=self.executor_spec.get('num_gpus', None)
        )

    def create_remote_workers(self, cls, num_actors, agent_config, worker_spec, *args):
        """
        Creates Ray actors for remote execution.

        Args:
            cls (RayWorker): Actor class, must be an instance of RayWorker.
            num_actors (int): Num
            agent_config (dict): Agent config.
            worker_spec (dict): Worker spec.
            *args (any): Arguments for RayWorker class.

        Returns:
            list: Remote Ray actors.
        """
        workers = []
        init_tasks = []

        cls_as_remote = cls.as_remote(num_cpus=self.num_cpus_per_worker, num_gpus=self.num_gpus_per_worker).remote

        # Create remote objects and schedule init tasks.
        ray_constant_exploration = worker_spec.get("ray_constant_exploration", False)
        for i in range_(num_actors):
            if ray_constant_exploration is True:
                exploration_val = worker_exploration(i, num_actors)
                worker_spec["ray_exploration"] = exploration_val
            worker = cls_as_remote(deepcopy(agent_config), worker_spec, *args)
            self.worker_ids[worker] = "worker_{}".format(i)
            workers.append(worker)
            build_result = worker.init_agent.remote()
            init_tasks.append(build_result)

        ready, not_ready = ray.wait(init_tasks, num_returns=len(init_tasks))

        for i, res in enumerate(ready):
            result = ray.get(res)
            if result:
                self.logger.info("Successfully built agent num {}.".format(i))

        return workers

    def setup_execution(self):
        """
        Creates and initializes all remote agents on the Ray cluster. Does not
        schedule any tasks yet.
        """
        raise NotImplementedError

    def init_tasks(self):
        """
        Initializes Remote ray worker tasks. Calling this method will result in
        actually scheduling tasks on Ray, as opposed to setup_execution which just
        creates the relevant remote actors.
        """
        pass

    def execute_workload(self, workload):
        """
        Executes a workload on Ray and measures worker statistics. Workload semantics
        are decided via the private implementer, _execute_step().
        Args:
            workload (dict): Workload parameters, primarily 'num_timesteps' and 'report_interval'
                to indicate how many steps to execute and how often to report results.
        """
        self.sample_iteration_throughputs = list()
        self.update_iteration_throughputs = list()
        self.iteration_times = list()

        # Assume time step based initially.
        num_timesteps = workload["num_timesteps"]

        # Performance reporting granularity.
        report_interval = workload["report_interval"]
        report_interval_min_seconds = workload["report_interval_min_seconds"]
        timesteps_executed = 0
        iteration_time_steps = list()
        iteration_update_steps = list()

        start = time.monotonic()
        # Call _execute_step as many times as required.
        while timesteps_executed < num_timesteps:
            iteration_step = 0
            iteration_updates = 0
            iteration_start = time.monotonic()

            # Record sampling and learning throughput every interval.
            while (iteration_step < report_interval) or\
                    time.monotonic() - iteration_start < report_interval_min_seconds:
                worker_steps_executed, update_steps = self._execute_step()
                iteration_step += worker_steps_executed
                iteration_updates += update_steps

            iteration_end = time.monotonic() - iteration_start
            timesteps_executed += iteration_step

            # Append raw values, compute stats after experiment is done.
            self.iteration_times.append(iteration_end)
            iteration_update_steps.append(iteration_updates)
            iteration_time_steps.append(iteration_step)

            self.logger.info("Executed {} Ray worker steps, {} update steps, ({} of {} ({} %))".format(
                iteration_step, iteration_updates, timesteps_executed,
                num_timesteps, (100 * timesteps_executed / num_timesteps)
            ))

        total_time = (time.monotonic() - start) or 1e-10
        self.logger.info("Time steps executed: {} ({} ops/s)".
                         format(timesteps_executed, timesteps_executed / total_time))
        all_updates = np.sum(iteration_update_steps)
        self.logger.info("Updates executed: {}, ({} updates/s)".format(
            all_updates, all_updates / total_time
        ))
        for i in range_(len(self.iteration_times)):
            it_time = self.iteration_times[i]
            # Note: these are samples, not internal environment frames.
            self.sample_iteration_throughputs.append(iteration_time_steps[i] / it_time)
            self.update_iteration_throughputs.append(iteration_update_steps[i] / it_time)

        worker_stats = self.get_aggregate_worker_results()
        self.logger.info("Retrieved worker stats for {} workers:".format(len(self.ray_env_sample_workers)))
        self.logger.info(worker_stats)

        return dict(
            runtime=total_time,
            timesteps_executed=timesteps_executed,
            ops_per_second=(timesteps_executed / total_time),
            # Multiply sample throughput by these env_frames = samples * env_internal * worker_frame_skip:
            env_internal_frame_skip=self.env_internal_frame_skip,
            worker_frame_skip=self.worker_frameskip,
            min_iteration_sample_throughput=np.min(self.sample_iteration_throughputs),
            max_iteration_sample_throughput=np.max(self.sample_iteration_throughputs),
            mean_iteration_sample_throughput=np.mean(self.sample_iteration_throughputs),
            min_iteration_update_throughput=np.min(self.update_iteration_throughputs),
            max_iteration_update_throughput=np.max(self.update_iteration_throughputs),
            mean_iteration_update_throughput=np.mean(self.update_iteration_throughputs),
            # Worker stats.
            mean_worker_op_throughput=worker_stats["mean_worker_op_throughput"],
            # N.b. these are already corrected.
            mean_worker_env_frames_throughput=worker_stats["mean_worker_env_frame_throughput"],
            max_worker_op_throughput=worker_stats["max_worker_op_throughput"],
            min_worker_op_throughput=worker_stats["min_worker_op_throughput"],
            mean_worker_reward=worker_stats["mean_reward"],
            max_worker_reward=worker_stats["max_reward"],
            min_worker_reward=worker_stats["min_reward"],
            # This is the mean final episode over all workers.
            mean_final_reward=worker_stats["mean_final_reward"]
        )

    def sample_metrics(self):
        return self.sample_iteration_throughputs

    def update_metrics(self):
        return self.update_iteration_throughputs

    def get_iteration_times(self):
        return self.iteration_times

    def _execute_step(self):
        """
        Actual private implementer of each step of the workload executed.
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
            Agent: RLGraph agent object.
        """
        config = deepcopy(agent_config)
        # Pop type on a copy because this may be called by multiple classes/worker types.
        agent_cls = Agent.__lookup_classes__.get(config.pop('type'))
        return agent_cls(**config)

    @staticmethod
    def build_env_from_config(env_spec):
        """
        Builds environment without using from_spec as Ray cannot handle kwargs correctly
        at the moment.

        Args:
            env_spec (dict): Environment specification. Must contain 'type' field to lookup constructor.

        Returns:
            Environment: Env object.
        """
        env_spec = deepcopy(env_spec)
        env_cls = Environment.lookup_class(env_spec.pop("type"))
        return env_cls(**env_spec)

    def result_by_worker(self, worker_index=None):
        """
        Retreives full episode-reward time series for a worker by id (or first worker in registry if None).

        Args:
            worker_index (Optional[int]): Index of worker to fetch.

        Returns:
            dict: Full results for this worker.
        """
        if worker_index is not None:
            ray_worker = self.ray_env_sample_workers[worker_index]
        else:
            # Otherwise just pick  first.
            ray_worker = self.ray_env_sample_workers[0]

        task = ray_worker.get_workload_statistics.remote()
        metrics = ray.get(task)

        # Return full reward series.
        return dict(
            episode_rewards=metrics["episode_rewards"],
            episode_timesteps=metrics["episode_timesteps"]
        )

    def get_all_worker_results(self):
        """
        Retrieves full episode-reward time series for all workers.

        Returns:
            list: List dicts with worker results (timesteps and rewards)
        """
        results = list()
        for ray_worker in self.ray_env_sample_workers:
            task = ray_worker.get_workload_statistics.remote()
            metrics = ray.get(task)
            results.append(dict(
                episode_rewards=metrics["episode_rewards"],
                episode_timesteps=metrics["episode_timesteps"],
                episode_total_times=metrics["episode_total_times"],
                episode_sample_times=metrics["episode_sample_times"]
            ))
        return results

    def get_sample_worker_ids(self):
        """
        Returns identifeirs of all sample workers.

        Returns:
            list: List of worker name strings in case individual analysis of one worker's results are required via
                'result_by_worker'.
        """
        return list(self.worker_ids.keys())

    def get_aggregate_worker_results(self):
        """
        Fetches execution statistics from remote workers and aggregates them.

        Returns:
            dict: Aggregate worker statistics.
        """
        min_rewards = list()
        max_rewards = list()
        mean_rewards = list()
        final_rewards = list()
        worker_op_throughputs = list()
        worker_env_frame_throughputs = list()
        episodes_executed = list()
        steps_executed = 0

        for ray_worker in self.ray_env_sample_workers:
            self.logger.info("Retrieving workload statistics for worker: {}".format(
                self.worker_ids[ray_worker])
            )
            task = ray_worker.get_workload_statistics.remote()
            metrics = ray.get(task)
            if metrics["mean_episode_reward"] is not None:
                min_rewards.append(metrics["min_episode_reward"])
                max_rewards.append(metrics["max_episode_reward"])
                mean_rewards.append(metrics["mean_episode_reward"])
                final_rewards.append(metrics["final_episode_reward"])
            else:
                self.logger.warning("Warning: No episode rewards available for worker {}. Steps executed: {}".
                                    format(self.worker_ids[ray_worker], metrics["worker_steps"]))
            episodes_executed.append(metrics["episodes_executed"])
            steps_executed += metrics["worker_steps"]
            worker_op_throughputs.append(metrics["mean_worker_ops_per_second"])
            worker_env_frame_throughputs.append(metrics["mean_worker_env_frames_per_second"])

        return dict(
            min_reward=np.min(min_rewards),
            max_reward=np.max(max_rewards),
            mean_reward=np.mean(mean_rewards),
            mean_final_reward=np.mean(final_rewards),
            min_worker_episodes=np.min(episodes_executed),
            max_worker_episodes=np.max(episodes_executed),
            mean_worker_episodes=np.mean(episodes_executed),
            total_episodes_executed=np.sum(episodes_executed),
            steps_executed=steps_executed,
            # Identify potential straggling workers.
            mean_worker_op_throughput=np.mean(worker_op_throughputs),
            min_worker_op_throughput=np.min(worker_op_throughputs),
            max_worker_op_throughput=np.max(worker_op_throughputs),
            mean_worker_env_frame_throughput=np.mean(worker_env_frame_throughputs)
        )
