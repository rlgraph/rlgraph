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

from yarl.agents import Agent
from yarl.execution.ray.ray_executor import RayExecutor
from threading import Thread
from six.moves import queue
import time

import ray

from yarl.execution.ray.ray_util import create_colocated_agents, RayTaskPool


class ApexExecutor(RayExecutor):
    """
    Implements the distributed update semantics of distributed prioritized experience replay (APE-X),
    as described in:

    https://arxiv.org/abs/1803.00933
    """

    def __init__(self, agent_config, environment_id, cluster_spec):
        """

        Args:
            config (dict): Config dict containing agent and execution specs.
            environment_id (str): Environment identifier. Each worker in the cluster will instantiate
                an environment using this id.
        """
        super(ApexExecutor, self).__init__(cluster_spec)
        self.config = agent_config

        # Must specify an agent type.
        assert "type" in self.config
        self.environment_id = environment_id

        # These are the Ray remote tasks which sample batches from the replay memory
        # and pass them to the learner.
        self.prioritized_replay_tasks = RayTaskPool()

        # These are the tasks actually interacting with the environment.
        self.env_sample_tasks = RayTaskPool()
        self.replay_sampling_task_depth = self.cluster_spec['task_queue_depth']
        self.env_interaction_task_depth = self.cluster_spec['env_interaction_task_depth']

    def setup_execution(self):
        # Start Ray.
        self.ray_init()

        # Setup queues for communication between main communication loop and learner.
        self.sample_input_queue = queue.Queue(maxsize=self.cluster_spec['learn_queue_size'])
        self.update_output_queue = queue.Queue()

        # Create local worker agent according to spec.
        self.local_agent = Agent.from_spec(self.config)

        # Set up worker thread for performing updates.
        self.update_worker = UpdateWorker(
            agent=self.local_agent,
            input_queue=self.sample_input_queue,
            output_queue=self.update_output_queue
        )

        # Create remote sample workers based on ray cluster spec.
        self.num_sample_workers = self.cluster_spec['num_workers']
        self.ray_replay_agents = create_colocated_agents(
            agent_config=self.config,
            num_agents=self.num_sample_workers
        )

        self.env_sample_agents = None

    def init_tasks(self):
        """
        Triggers Remote ray tasks.
        """

        # Prioritized replay sampling tasks.
        for ray_agent in self.ray_replay_agents:
            for _ in range(self.replay_sampling_task_depth):
                # This initializes remote tasks to sample from the prioritized replay memories of each worker.
                self.prioritized_replay_tasks.add_task(ray_agent, ray_agent.get_batch.remote())

        # Env interaction tasks.
        for ray_agent in self.env_sample_agents:
            # TODO create wrapper for local env workers?
            pass

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

        return dict(
            runtime=total_time,
            timesteps_executed=timesteps_executed,
        )

    def _execute_step(self):
        """
        Performs a single step on the distributed Ray execution.
        """
        # TODO Iterate over sample tasks
        env_steps = 0
        # 2. Fetch completed replay priority sampling task, move to worker, reschedule.
        for ray_worker, replay_remote_task in self.prioritized_replay_tasks.get_completed():
            # Immediately schedule new batch sampling tasks on these workers.
            self.prioritized_replay_tasks.add_task(ray_worker, ray_worker.get_batch.remote())

            # Retrieve results via id.
            sampled_batch = ray.get(object_ids=replay_remote_task)

            # Pass to the agent doing the actual updates.
            # The ray worker is passed along because we need to update its priorities later in the subsequent
            # task (see loop below).
            self.sample_input_queue.put((ray_worker, sampled_batch))

        # 3. Update priorities on priority sampling workers using loss values produced by update worker.
        while not self.update_output_queue.empty():
            ray_worker, sampled_batch, loss = self.update_output_queue.get()

            # Use generic graph call op.
            ray_worker.call_graph_op.remote("update_priorities", [sampled_batch, loss])
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

    def run(self):
        while True:
            # Fetch input for update, update
            agent, sample_batch = self.input_queue.get()

            if sample_batch is not None:
                loss = self.agent.update(batch=sample_batch)
                self.output_queue.put((agent, sample_batch, loss))
