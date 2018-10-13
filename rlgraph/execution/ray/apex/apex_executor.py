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

import time
import random
from six.moves import queue
from threading import Thread

from rlgraph import get_distributed_backend
from rlgraph.agents import Agent
from rlgraph.execution.ray import RayWorker
from rlgraph.execution.ray.apex.ray_memory_actor import RayMemoryActor
from rlgraph.execution.ray.ray_executor import RayExecutor
from rlgraph.execution.ray.ray_util import create_colocated_ray_actors, RayTaskPool

if get_distributed_backend() == "ray":
    import ray


class ApexExecutor(RayExecutor):
    """
    Implements the distributed update semantics of distributed prioritized experience replay (Ape-X),
    as described in:

    https://arxiv.org/abs/1803.00933
    """
    def __init__(self, environment_spec, agent_config):
        """
        Args:
            environment_spec (dict): Environment spec. Each worker in the cluster will instantiate
                an environment using this spec.
            agent_config (dict): Config dict containing agent and execution specs.
        """
        ray_spec = agent_config["execution_spec"].pop("ray_spec")
        self.apex_replay_spec = ray_spec.pop("apex_replay_spec")
        self.worker_spec = ray_spec.pop("worker_spec")
        super(ApexExecutor, self).__init__(executor_spec=ray_spec.pop("executor_spec"),
                                           environment_spec=environment_spec,
                                           worker_spec=self.worker_spec)

        # Must specify an agent type.
        assert "type" in agent_config
        self.agent_config = agent_config

        # These are the Ray remote tasks which sample batches from the replay memory
        # and pass them to the learner.
        self.prioritized_replay_tasks = RayTaskPool()
        self.replay_sampling_task_depth = self.executor_spec["replay_sampling_task_depth"]
        self.replay_batch_size = self.agent_config["update_spec"]["batch_size"]
        self.num_cpus_per_replay_actor = self.executor_spec.get("num_cpus_per_replay_actor",
                                                                self.replay_sampling_task_depth)

        # How often weights are synced to remote workers.
        self.weight_sync_steps = self.executor_spec["weight_sync_steps"]

        # Necessary for target network updates.
        self.weight_syncs_executed = 0
        self.steps_since_weights_synced = dict()

        # These are the tasks actually interacting with the environment.
        self.env_sample_tasks = RayTaskPool()
        self.env_interaction_task_depth = self.executor_spec["env_interaction_task_depth"]
        self.worker_sample_size = self.executor_spec["num_worker_samples"] + self.worker_spec["n_step_adjustment"] - 1

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

        self.local_agent = self.build_agent_from_config(self.agent_config)

        # Set up worker thread for performing updates.
        self.update_worker = UpdateWorker(
            agent=self.local_agent,
            in_queue_size=self.executor_spec["learn_queue_size"]
        )

        # Create remote sample workers based on ray cluster spec.
        self.num_replay_workers = self.executor_spec["num_replay_workers"]
        self.num_sample_workers = self.executor_spec["num_sample_workers"]

        self.logger.info("Initializing {} local replay memories.".format(self.num_replay_workers))
        # Update memory size for num of workers
        shard_size = int(self.apex_replay_spec["memory_spec"]["capacity"] / self.num_replay_workers)
        self.apex_replay_spec["memory_spec"]["capacity"] = shard_size
        self.logger.info("Shard size per memory: {}".format(self.apex_replay_spec["memory_spec"]["capacity"]))
        min_sample_size = self.apex_replay_spec["min_sample_memory_size"]
        self.apex_replay_spec["min_sample_memory_size"] = int(min_sample_size / self.num_replay_workers)
        self.logger.info("Sampling for learning starts at: {}".format( self.apex_replay_spec["min_sample_memory_size"]))

        # Set sample batch size:
        self.apex_replay_spec["sample_batch_size"] = self.agent_config["update_spec"]["batch_size"]
        self.logger.info("Sampling batch size {}".format(self.apex_replay_spec["sample_batch_size"]))

        self.ray_local_replay_memories = create_colocated_ray_actors(
            cls=RayMemoryActor.as_remote(num_cpus=self.num_cpus_per_replay_actor),
            config=self.apex_replay_spec,
            num_agents=self.num_replay_workers
        )

        # Create remote workers for data collection.
        self.worker_spec["worker_sample_size"] = self.worker_sample_size
        self.logger.info("Initializing {} remote data collection agents, sample size: {}".format(
            self.num_sample_workers, self.worker_spec["worker_sample_size"]))
        self.ray_env_sample_workers = self.create_remote_workers(
            RayWorker, self.num_sample_workers, self.agent_config,
            # *args
            self.worker_spec, self.environment_spec, self.worker_frameskip
        )
        self.init_tasks()

    def test_worker_init(self):
        """
        Tests every worker for successful constructor call (which may otherwise fail silently.
        """
        for ray_worker in self.ray_env_sample_workers:
            self.logger.info("Testing worker for successful init: {}".format(self.worker_ids[ray_worker]))
            task = ray_worker.get_constructor_success.remote()
            result = ray.get(task)
            assert result is True, "ERROR: constructor failed, attribute returned: {}" \
                                   "instead of True".format(result)

    def init_tasks(self):
        # Start learner thread.
        self.update_worker.start()

        # Prioritized replay sampling tasks via RayAgents.
        for ray_memory in self.ray_local_replay_memories:
            for _ in range(self.replay_sampling_task_depth):
                # This initializes remote tasks to sample from the prioritized replay memories of each worker.
                self.prioritized_replay_tasks.add_task(ray_memory, ray_memory.get_batch.remote())

        # Env interaction tasks via RayWorkers which each
        # have a local agent.
        weights = self.local_agent.get_policy_weights()
        for ray_worker in self.ray_env_sample_workers:
            ray_worker.set_policy_weights.remote(weights)
            self.steps_since_weights_synced[ray_worker] = 0

            self.logger.info("Synced worker {} weights, initializing sample tasks.".format(
                self.worker_ids[ray_worker]))
            for _ in range(self.env_interaction_task_depth):
                self.env_sample_tasks.add_task(ray_worker, ray_worker.execute_and_get_with_count.remote())

    def _execute_step(self):
        """
        Executes a workload on Ray. The main loop performs the following
        steps until the specified number of steps or episodes is finished:

        - Retrieve sample batches via Ray from remote workers
        - Insert these into the local memory
        - Have a separate learn thread sample batches from the memory and compute updates
        - Sync weights to the shared model so remot eworkers can update their weights.
        """
        # Env steps done during this rollout.
        env_steps = 0
        update_steps = 0
        weights = None

        # 1. Fetch results from RayWorkers.
        completed_sample_tasks = list(self.env_sample_tasks.get_completed())
        sample_batch_sizes = ray.get([task[1][1] for task in completed_sample_tasks])
        for i, (ray_worker, (env_sample_obj_id, sample_size)) in enumerate(completed_sample_tasks):
            # Randomly add env sample to a local replay actor.
            random.choice(self.ray_local_replay_memories).observe.remote(env_sample_obj_id)
            sample_steps = sample_batch_sizes[i]
            env_steps += sample_steps

            self.steps_since_weights_synced[ray_worker] += sample_steps
            if self.steps_since_weights_synced[ray_worker] >= self.weight_sync_steps:
                if weights is None or self.update_worker.update_done:
                    self.update_worker.update_done = False
                    weights = ray.put(self.local_agent.get_policy_weights())
                # self.logger.debug("Syncing weights for worker {}".format(self.worker_ids[ray_worker]))
                # self.logger.debug("Weights type: {}, weights = {}".format(type(weights), weights))
                ray_worker.set_policy_weights.remote(weights)
                self.weight_syncs_executed += 1
                self.steps_since_weights_synced[ray_worker] = 0

            # Reschedule environment samples.
            self.env_sample_tasks.add_task(ray_worker, ray_worker.execute_and_get_with_count.remote())

        # 2. Fetch completed replay priority sampling task, move to worker, reschedule.
        for ray_memory, replay_remote_task in self.prioritized_replay_tasks.get_completed():
            # Immediately schedule new batch sampling tasks on these workers.
            self.prioritized_replay_tasks.add_task(ray_memory, ray_memory.get_batch.remote())

            # Retrieve results via id.
            # self.logger.info("replay task obj id {}".format(replay_remote_task))
            sampled_batch = ray.get(object_ids=replay_remote_task)
            # if sampled_batch is not None:
            #     self.logger.info("Received result of replay task: {}".format(len(sampled_batch["terminals"])))
            #     self.logger.info("Received result of replay task: {}".format(len(sampled_batch["indices"])))

            # Pass to the agent doing the actual updates.
            # The ray worker is passed along because we need to update its priorities later in the subsequent
            # task (see loop below).
            self.update_worker.input_queue.put((ray_memory, sampled_batch))

        # 3. Update priorities on priority sampling workers using loss values produced by update worker.
        while not self.update_worker.output_queue.empty():
            ray_memory, indices, loss_per_item = self.update_worker.output_queue.get()
            # self.logger.info('indices = {}'.format(batch["indices"]))
            # self.logger.info('loss = {}'.format(loss_per_item))

            ray_memory.update_priorities.remote(indices, loss_per_item)
            # len of loss per item is update count.
            update_steps += len(indices)

        return env_steps, update_steps


class UpdateWorker(Thread):
    """
    Executes learning separate from the main event loop as described in the Ape-X paper.
    Communicates with the main thread via a queue.
    """

    def __init__(self, agent, in_queue_size):
        """
        Initializes the worker with a RLGraph agent and queues for

        Args:
            agent (Agent): RLGraph agent used to execute local updates.
            input_queue (queue.Queue): Input queue the worker will use to poll samples.
            output_queue (queue.Queue): Output queue the worker will use to push results of local
                update computations.
        """
        super(UpdateWorker, self).__init__()

        # Agent to use for updating.
        self.agent = agent
        self.input_queue = queue.Queue(maxsize=in_queue_size)
        self.output_queue = queue.Queue()

        # Terminate when host process terminates.
        self.daemon = True

        # Flag for main thread.
        self.update_done = False

    def run(self):
        while True:
            self.step()

    def step(self):
        # Fetch input for update:
        # Replay memory used.
        memory_actor, sample_batch = self.input_queue.get()

        if sample_batch is not None:
            loss, loss_per_item = self.agent.update(batch=sample_batch)
            # Just pass back indices for updating.
            self.output_queue.put((memory_actor, sample_batch["indices"], loss_per_item))
            self.update_done = True
