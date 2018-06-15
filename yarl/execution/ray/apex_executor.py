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

from six.moves import queue

from yarl import get_distributed_backend
from yarl.agents import Agent
from yarl.execution.ray import RayWorker
from yarl.execution.ray.ray_executor import RayExecutor
import random
from threading import Thread
import time

from yarl.execution.ray.ray_util import create_colocated_agents, RayTaskPool

if get_distributed_backend() == "ray":
    import ray


class ApexExecutor(RayExecutor):
    """
    Implements the distributed update semantics of distributed prioritized experience replay (APE-X),
    as described in:

    https://arxiv.org/abs/1803.00933
    """
    def __init__(self, environment_spec, agent_config, cluster_spec, repeat_actions=1):
        """
        Args:
            environment_spec (dict): Environment spec. Each worker in the cluster will instantiate
                an environment using this spec.
            agent_config (dict): Config dict containing agent and execution specs.
            # TODO this does not seem like it belongs here
            repeat_actions (Optional[int]): How often actions are repeated after retrieving them from the agent.

        """
        super(ApexExecutor, self).__init__(cluster_spec)
        self.environment_spec = environment_spec
        # Must specify an agent type.
        assert "type" in self.agent_config
        self.agent_config = agent_config
        self.repeat_actions = repeat_actions

        # These are the Ray remote tasks which sample batches from the replay memory
        # and pass them to the learner.
        self.prioritized_replay_tasks = RayTaskPool()
        self.replay_sampling_task_depth = self.cluster_spec['task_queue_depth']

        # How often weights are synced to remote workers.
        self.weight_sync_steps = self.cluster_spec['weight_sync_steps']
        self.steps_since_weights_synced = dict()

        # These are the tasks actually interacting with the environment.
        self.env_sample_tasks = RayTaskPool()
        self.env_interaction_task_depth = self.cluster_spec['env_interaction_task_depth']

        self.worker_sample_size = self.cluster_spec['num_worker_samples']

    def setup_execution(self):
        # Start Ray.
        self.ray_init()

        # Setup queues for communication between main communication loop and learner.
        self.sample_input_queue = queue.Queue(maxsize=self.cluster_spec['learn_queue_size'])
        self.update_output_queue = queue.Queue()

        # Create local worker agent according to spec.
        self.local_agent = Agent.from_spec(self.agent_config)

        # Set up worker thread for performing updates.
        self.update_worker = UpdateWorker(
            agent=self.local_agent,
            input_queue=self.sample_input_queue,
            output_queue=self.update_output_queue
        )

        # Create remote sample workers based on ray cluster spec.
        self.num_sample_workers = self.cluster_spec['num_workers']
        self.ray_replay_agents = create_colocated_agents(
            agent_config=self.agent_config,
            num_agents=self.num_sample_workers
        )

        # Create remote workers for data collection.
        self.ray_workers = self.create_remote_workers(
            RayWorker,
            self.num_sample_workers,
            [self.environment_spec, self.agent_config, self.repeat_actions]
        )

    def init_tasks(self):
        """
        Initializes Remote ray worker tasks.
        """

        # Prioritized replay sampling tasks via RayAgents.
        for ray_agent in self.ray_replay_agents:
            for _ in range(self.replay_sampling_task_depth):
                # This initializes remote tasks to sample from the prioritized replay memories of each worker.
                self.prioritized_replay_tasks.add_task(ray_agent, ray_agent.get_batch.remote())

        # Env interaction tasks via RayWorkers which each
        # have a local agent.
        weights = self.local_agent.get_weights()
        for ray_worker in self.ray_workers:
            self.steps_since_weights_synced[ray_worker] = 0
            ray_worker.set_weights.remote(weights)
            for _ in range(self.env_interaction_task_depth):
                self.env_sample_tasks.add_task(ray_worker, ray_worker.execute_and_get_timesteps.remote(
                    num_timesteps=self.worker_sample_size,
                    break_on_terminal=True
                ))

    def execute_workload(self, workload):
        """
        Executes a workload via Ape-X semantics. The main loop performs the following
        steps until the specified number of steps or episodes is finished:

        - Retrieve sample batches via Ray from remote workers
        - Insert these into the local memory
        - Have a separate learn thread sample batches from the memory and compute updates
        - Sync weights to the shared model so remot eworkers can update their weights.
        """
        start = time.monotonic()
        self.init_tasks()

        # Assume time step based initially.
        num_timesteps = workload['num_timesteps']
        timesteps_executed = 0
        step_times = []

        # Call _execute_step as many times as required.
        while timesteps_executed < num_timesteps:
            step_time = time.monotonic()
            executed = self._execute_step()
            step_times.append(time.monotonic() - step_time)
            timesteps_executed += executed

        total_time = (time.monotonic() - start) or 1e-10
        self.logger.info("Time steps (actions) executed: {} ({} ops/s)".
                         format(timesteps_executed, timesteps_executed / total_time))

        # TODO add more stats from steps
        return dict(
            runtime=total_time,
            timesteps_executed=timesteps_executed,
        )

    def _execute_step(self):
        """
        Performs a single step on the distributed Ray execution.
        """
        # Env steps done during this rollout.
        env_steps = 0
        weights = None
        # 1. Fetch results from RayWorkers.
        for ray_worker, env_sample in self.env_sample_tasks.get_completed():
            # Randomly add env sample to a local replay actor.
            random_actor = random.choice(self.ray_replay_agents)
            sample_data = env_sample.get_batch()

            # TODO: Check if we should break on terminal, in that case this here overestimates num of frames
            env_steps += self.worker_sample_size
            random_actor.observe.remote(
                states=sample_data['states'],
                actions=sample_data['actions'],
                internals=None,
                rewards=sample_data['rewards'],
                terminal=sample_data['terminal']
            )

            self.steps_since_weights_synced[ray_worker] += self.worker_sample_size
            if self.steps_since_weights_synced[ray_worker] >= self.weight_sync_steps:
                if weights is None or self.update_worker.update_done:
                    self.update_worker.update_done = False
                    weights = ray.put(self.local_agent.get_weights())
                ray_worker.set_weights.remote(weights)
                self.steps_since_weights_synced[ray_worker] = 0

            # Reschedule environment samples.
            self.env_sample_tasks.add_task(ray_worker, ray_worker.remote.execute_and_get_timesteps(
                num_timesteps=self.worker_sample_size,
                break_on_terminal=True
            ))

        # 2. Fetch completed replay priority sampling task, move to worker, reschedule.
        for ray_agent, replay_remote_task in self.prioritized_replay_tasks.get_completed():
            # Immediately schedule new batch sampling tasks on these workers.
            self.prioritized_replay_tasks.add_task(ray_agent, ray_agent.get_batch.remote())

            # Retrieve results via id.
            sampled_batch = ray.get(object_ids=replay_remote_task)

            # Pass to the agent doing the actual updates.
            # The ray worker is passed along because we need to update its priorities later in the subsequent
            # task (see loop below).
            self.sample_input_queue.put((ray_agent, sampled_batch))

        # 3. Update priorities on priority sampling workers using loss values produced by update worker.
        while not self.update_output_queue.empty():
            ray_agent, sampled_batch, loss = self.update_output_queue.get()

            # Use generic graph call op.
            ray_agent.call_graph_op.remote("update_priorities", [sampled_batch, loss])
        return env_steps


class UpdateWorker(Thread):
    """
    Executes learning separate from the main event loop as described in the Ape-X paper.
    Communicates with the main thread via a queue.
    """

    def __init__(self, agent, input_queue, output_queue):
        """
        Initializes the worker with a YARL agent and queues for

        Args:
            agent (Agent): YARL agent used to execute local updates.
            input_queue (queue.Queue): Input queue the worker will use to poll samples.
            output_queue (queue.Queue): Output queue the worker will use to push results of local
                update computations.
        """
        super(UpdateWorker, self).__init__()

        # Agent to use for updating.
        self.agent = agent
        self.input_queue = input_queue
        self.output_queue = output_queue

        # Terminate when host process terminates.
        self.daemon = True

        # Flag for main thread.
        self.update_done = False

    def run(self):
        while True:
            # Fetch input for update, update
            agent, sample_batch = self.input_queue.get()

            if sample_batch is not None:
                loss = self.agent.update(batch=sample_batch)
                self.output_queue.put((agent, sample_batch, loss))
                self.update_done = True

