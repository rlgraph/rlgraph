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

import logging
from six.moves import xrange as range_


class Worker(object):
    """
    Generic worker to locally interact with simulator environments.
    """

    def __init__(self, environment, agent, repeat_actions=1):
        """
        Initializes an executor
        Args:
            environment (env): Environment to execute.
            agent (Agent): Agent to execute environment on.
            repeat_actions (int): How often actions are repeated after retrieving them from the agent.
        """
        self.logger = logging.getLogger(__name__)
        self.environment = environment
        self.agent = agent
        self.repeat_actions = repeat_actions

        # Update schedule if worker is performing updates.
        #self.updating = False
        #self.update_interval = None
        #self.update_steps = None
        #self.steps_before_update = None

    def execute_timesteps(self, num_timesteps, deterministic, update_schedule=None):
        """
        Executes environment for a fixed number of timesteps.

        Args:
            num_timesteps (int): Number of time steps to execute.
            deterministic (bool): Indicates deterministic execution.
            update_schedule (Optional[dict]): Update parameters. If None, the worker only peforms rollouts.
                Expects keys 'update_interval' to indicate how frequent update is called, 'num_updates'
                to indicate how many updates to perform every update interval, and 'steps_before_update' to indicate
                how many steps to perform before beginning to update.
        Returns:
            dict: Execution statistics.
        """
        raise NotImplementedError

    def execute_episodes(self, num_episodes, max_timesteps_per_episode, update_schedule=None):
        """
        Executes environment for a fixed number of episodes.

        Args:
            num_episodes (int): Number of episodes to execute.
            max_timesteps_per_episode: Maximum length per episode.
            deterministic (bool): Indicates deterministic execution.
            update_schedule (Optional[dict]): Update parameters. If None, the worker only peforms rollouts.
                Expects keys 'update_interval' to indicate how frequent update is called, 'num_updates'
                to indicate how many updates to perform every update interval, and 'steps_before_update' to indicate
                how many steps to perform before beginning to update.
        Returns:
            dict: Execution statistics.
        """
        raise NotImplementedError

    def update_if_necessary(self, timesteps_executed, update_spec):
        """
        Calls update on the agent according to the update schedule set for this worker.

        Args:
            timesteps_executed (int): Timesteps executed thus far.
            update_spec (dict): The update_spec spec-dict.
        """
        if update_spec["do_updates"] is True:
            if timesteps_executed > update_spec["steps_before_update"] and \
                    timesteps_executed % update_spec["update_interval"] == 0:
                for _ in range_(update_spec["update_steps"]):
                    self.agent.update()

