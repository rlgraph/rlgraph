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


class Executor(object):
    """
    Generic executor to run simulator environments..
    """

    def __init__(self, environment, agent):
        """
        Initializes an executor
        Args:
            environment (env):
            agent (Agent):
        """
        self.environment = environment
        self.agent = agent

    def execute_timesteps(self, num_timesteps):
        """
        Executes environment for a fixed number of timesteps.
        Args:
            num_timesteps (int): Number of time steps to execute.
        """
        raise NotImplementedError

    def execute_episodes(self, num_episodes, max_timesteps_per_episode):
        """
        Executes environment for a fixed number of episodes.

        Args:
            num_episodes (int): Number of episodes to execute.
            max_timesteps_per_episode: Maximum length per episode.
        """
        raise NotImplementedError
