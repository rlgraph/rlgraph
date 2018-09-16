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

import numpy as np
import logging
from six.moves import xrange as range_

from rlgraph.utils.specifiable import Specifiable
from rlgraph.environments import SequentialVectorEnv


class Worker(Specifiable):
    """
    Generic worker to locally interact with simulator environments.
    """
    def __init__(self, agent, env_spec=None, num_envs=1, frameskip=1, render=False,
                 worker_executes_exploration=True, exploration_epsilon=0.1):
        """
        Args:
            agent (Agent): Agent to execute environment on.
            env_spec Optional[Union[callable, dict]]): Either an environment spec or a callable returning a new
                environment.
            num_envs (int): How many single Environments should be run in parallel in a SequentialVectorEnv.
            frameskip (int): How often actions are repeated after retrieving them from the agent.
                This setting can be overwritten in the single calls to the different `execute_..` methods.
            render (bool): Whether to render the environment after each action.
                Default: False.
            worker_executes_exploration (bool): If worker executes exploration by sampling.
            exploration_epsilon (Optional[float]): Epsilon to use if worker executes exploration.
        """
        super(Worker, self).__init__()
        self.num_environments = num_envs
        self.logger = logging.getLogger(__name__)
        if env_spec is not None:
            self.env_ids = ["env_{}".format(i) for i in range_(self.num_environments)]
            self.vector_env = SequentialVectorEnv(env_spec=env_spec, num_envs=self.num_environments)
        else:
            self.env_ids = []
            self.vector_env = None
        self.agent = agent
        self.frameskip = frameskip
        self.render = render

        # Update schedule if worker is performing updates.
        self.updating = None
        self.steps_before_update = None
        self.update_interval = None
        self.update_steps = None
        self.sync_interval = None

        self.worker_executes_exploration = worker_executes_exploration
        self.exploration_epsilon = exploration_epsilon

    def execute_timesteps(self, num_timesteps, max_timesteps_per_episode=0, update_spec=None, use_exploration=True,
                          frameskip=1, reset=True):
        """
        Executes environment for a fixed number of timesteps.

        Args:
            num_timesteps (int): Number of time steps to execute.
            max_timesteps_per_episode (Optional[int]): Can be used to limit the number of timesteps per episode.
                Use None or 0 for no limit. Default: None.
            update_spec (Optional[dict]): Update parameters. If None, the worker only peforms rollouts.
                Expects keys 'update_interval' to indicate how frequent update is called, 'num_updates'
                to indicate how many updates to perform every update interval, and 'steps_before_update' to indicate
                how many steps to perform before beginning to update.
            use_exploration (Optional[bool]): Indicates whether to utilize exploration (epsilon or noise based)
                when picking actions.
            frameskip (int): How often actions are repeated after retrieving them from the agent.
                Use None for the Worker's default value.
            reset (bool): Whether to reset the environment and all the Worker's internal counters.
                Default: True.

        Returns:
            dict: Execution statistics.
        """
        pass

    def execute_and_get_timesteps(self, num_timesteps, max_timesteps_per_episode=0, use_exploration=True,
                                  frameskip=None, reset=True):
        """
        Executes timesteps and returns experiences. Intended for distributed data collection
        without performing updates.

        Args:
            num_timesteps (int): Number of time steps to execute.
            max_timesteps_per_episode (Optional[int]): Can be used to limit the number of timesteps per episode.
                Use None or 0 for no limit. Default: None.
            use_exploration (Optional[bool]): Indicates whether to utilize exploration (epsilon or noise based)
                when picking actions.
            frameskip (int): How often actions are repeated after retrieving them from the agent.
                Use None for the Worker's default value.
            reset (bool): Whether to reset the environment and all the Worker's internal counters.
                Default: True.

        Returns:
            EnvSample: EnvSample object holding the collected experiences.
        """
        pass

    def execute_episodes(self, num_episodes, max_timesteps_per_episode=0, update_spec=None, use_exploration=True,
                         frameskip=None, reset=True):
        """
        Executes environment for a fixed number of episodes.

        Args:
            num_episodes (int): Number of episodes to execute.
            max_timesteps_per_episode (Optional[int]): Can be used to limit the number of timesteps per episode.
                Use None or 0 for no limit. Default: None.
            update_spec (Optional[dict]): Update parameters. If None, the worker only peforms rollouts.
                Expects keys 'update_interval' to indicate how frequent update is called, 'num_updates'
                to indicate how many updates to perform every update interval, and 'steps_before_update' to indicate
                how many steps to perform before beginning to update.
            use_exploration (Optional[bool]): Indicates whether to utilize exploration (epsilon or noise based)
                when picking actions.
            frameskip (int): How often actions are repeated after retrieving them from the agent.
                Use None for the Worker's default value.
            reset (bool): Whether to reset the environment and all the Worker's internal counters.
                Default: True.

        Returns:
            dict: Execution statistics.
        """
        pass

    def execute_and_get_episodes(self, num_episodes, max_timesteps_per_episode=0, use_exploration=False,
                                 frameskip=None, reset=True):
        """
        Executes episodes and returns experiences as separate episode sequences.
        Intended for distributed data collection without performing updates.

        Args:
            num_episodes (int): Number of episodes to execute.
            max_timesteps_per_episode (Optional[int]): Can be used to limit the number of timesteps per episode.
                Use None or 0 for no limit. Default: None.
            use_exploration (Optional[bool]): Indicates whether to utilize exploration (epsilon or noise based)
                when picking actions.
            frameskip (int): How often actions are repeated after retrieving them from the agent. Rewards are
                accumulated over the skipped frames.
            reset (bool): Whether to reset the environment and all the Worker's internal counters.
                Default: True.

        Returns:
            EnvSample: EnvSample object holding the collected episodes.
        """
        pass

    def update_if_necessary(self):
        """
        Calls update on the agent according to the update schedule set for this worker.

        #Args:
        #    timesteps_executed (int): Timesteps executed thus far.

        Returns:
            float: The summed up loss (over all self.update_steps).
        """
        if self.updating:
            # Are we allowed to update?
            if self.agent.timesteps > self.steps_before_update and \
                    (self.agent.observe_spec["buffer_enabled"] is False or  # no update before some data in buffer
                     self.agent.timesteps >= self.agent.observe_spec["buffer_size"]) and \
                    self.agent.timesteps % self.update_interval == 0:  # update frequency check
                loss = 0
                for _ in range_(self.update_steps):
                    ret = self.agent.update()
                    if isinstance(ret, tuple):
                        loss += ret[0]
                    else:
                        loss += ret
                return loss

        return None

    def set_update_schedule(self, update_schedule=None):
        """
        Sets this worker's update schedule. By default, a worker is not updating but only acting
        and observing samples.

        Args:
            update_schedule (Optional[dict]): Update parameters. If None, the worker only performs rollouts.
                Expects keys 'update_interval' to indicate how frequent update is called, 'num_updates'
                to indicate how many updates to perform every update interval, and 'steps_before_update' to indicate
                how many steps to perform before beginning to update.
        """
        if update_schedule is not None:
            self.updating = True
            self.steps_before_update = update_schedule['steps_before_update']
            self.update_interval = update_schedule['update_interval']
            self.update_steps = update_schedule['update_steps']
            self.sync_interval = update_schedule['sync_interval']

    def get_action(self, states, use_exploration, apply_preprocessing, extra_returns):
        if self.worker_executes_exploration:
            # Only once for all actions otherwise we would have to call a session anyway.
            if np.random.random() <= self.exploration_epsilon:
                if self.num_environments == 1:
                    # Sample returns without batch dim -> wrap.
                    action = [self.agent.action_space.sample(size=self.num_environments)]
                else:
                    action = self.agent.action_space.sample(size=self.num_environments)
            else:
                if self.num_environments == 1:
                    action = [self.agent.get_action(states=states, use_exploration=use_exploration,
                              apply_preprocessing=apply_preprocessing, extra_returns=extra_returns)]
                else:
                    action = self.agent.get_action(states=states, use_exploration=use_exploration,
                             apply_preprocessing=apply_preprocessing, extra_returns=extra_returns)
            return action
        else:
            return self.agent.get_action(states=states, use_exploration=use_exploration,
                                         apply_preprocessing=apply_preprocessing, extra_returns=extra_returns)


