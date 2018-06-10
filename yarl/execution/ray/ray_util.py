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

import os

import logging
from yarl import YARLError
from yarl.execution.ray import RayAgent

import ray


# Largely follow utils used in Ray Rllib ported to use the RayAgent wrapper.


class RayTaskPool(object):
    """
    Manages a set of Ray tasks currently being executed (i.e. the RayAgent tasks).
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.ray_tasks = dict()

    def add_task(self, worker, ray_object_id):
        """
        Adds a task to the task pool.
        Args:
            worker (any): Worker completing the task, must use the @ray.remote decorator.
            ray_object_id (str): Ray object id. See ray documentation for how these are used.
        """
        self.logger.debug("Adding task {} to worker {}".format(
            ray_object_id, worker
        ))
        # Map which worker is responsible for completing the Ray task.
        self.ray_tasks[ray_object_id] = worker

    def get_completed(self):
        """
        Waits on pending tasks and yields them upon completion.

        Returns:
            generator: Yields completed tasks.
        """

        pending_tasks = list(self.ray_tasks)
        if pending_tasks:
            # This ray function checks tasks and splits into ready and non-ready tasks.
            ready, not_ready = ray.wait(pending_tasks, num_returns=len(pending_tasks), timeout=10)
            for obj_id in ready:
                yield (self.ray_tasks.pop(obj_id), obj_id)


def create_colocated_agents(agent_config, num_agents, max_attempts=10):
    """
    Creates a specified number of co-located RayAgent workers.

    Args:
        agent_config (dict): Agent spec for worker agents.
        num_agents (int): Number of worker agents to create.
        max_attempts Optional[int]: Max number of attempts to create colocated agents, will raise
            an error if creation was not successful within this number.

    Returns:
        list: List of created agents.

    Raises:
        YARL-Error if not enough agents could be created within the specified number of attempts.
    """
    agents = []
    attempt = 1

    while len(agents) < num_agents and attempt <= max_attempts:
        ray_agents = [RayAgent.remote(agent_config) for _ in range(attempt * num_agents)]
        local_agents, _ = split_local_non_local_agents(ray_agents)
        agents.extend(local_agents)

    if len(agents) < num_agents:
        raise YARLError("Could not create the specified number ({}) of agents.".format(
            num_agents
        ))

    return agents[:num_agents]


def split_local_non_local_agents(ray_agents):
    """
    Splits agents in local and non-local agents based on localhost string and ray remote
    hsots.

    Args:
        ray_agents (list): List of RayAgent objects.

    Returns:
        (list, list): Local and non-local agents.
    """
    localhost = os.uname()[1]
    hosts = ray.get([agent.get_host.remote() for agent in ray_agents])
    local = []
    non_local = []

    for host, a in zip(hosts, ray_agents):
        if host == localhost:
            local.append(a)
        else:
            non_local.append(a)
    return local, non_local
