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

# Largely follows utils used in Ray.
from yarl import YARLError


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
        pass

    if len(agents) < num_agents:
        raise YARLError("Could not create the specified number ({}) of agents.".format(
            num_agents
        ))


def split_local_non_local_agents(agents):
    """
    Splits agents in local and non-local agents based on localhost string and ray remote
    hsots.

    Args:
        agents (list): List of RayAgent objects.

    Returns:
        (list, list): Local and non-local agents.
    """
    pass