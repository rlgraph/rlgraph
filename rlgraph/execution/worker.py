# Copyright 2018/2019 The RLgraph authors. All Rights Reserved.
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

import numpy as np
from six.moves import xrange as range_

from rlgraph.environments import VectorEnv, SequentialVectorEnv
from rlgraph.execution.rules.update_rules import UpdateRules
from rlgraph.utils.specifiable import Specifiable


class Worker(Specifiable):
    """
    Generic worker to locally interact with simulator environments.
    """
    def __init__(self, agent, env_spec=None, update_rules=None, num_environments=1, frameskip=1, render=False,
                 worker_executes_exploration=True, exploration_epsilon=0.1, episode_finish_callback=None,
                 update_finish_callback=None, max_timesteps=None):
        """
        Args:
            agent (Agent): Agent to execute environment on.

            env_spec Optional[Union[callable, dict]]): Either an environment spec or a callable returning a new
                environment.

            update_rules (Optional[dict,UpdateRules]): Specification dict (or UpdateRules object directly) to construct
                the UpdateRules object that this Worker will use to schedule `Agent.update()` calls.

            num_environments (int): How many single Environments should be run in parallel in a SequentialVectorEnv.

            frameskip (int): How often actions are repeated after retrieving them from the agent.
                This setting can be overwritten in the single calls to the different `execute_..` methods.

            render (bool): Whether to render the environment after each action.
                Default: False.

            worker_executes_exploration (bool): If worker executes exploration by sampling.
            exploration_epsilon (Optional[float]): Epsilon to use if worker executes exploration.
            episode_finish_callback (Optional[callable]): An optional callback function to print out episode stats.
            update_finish_callback (Optional[callable]): An optional callback function to print out update stats.

            max_timesteps (Optional[int]): A max number on the time steps this Worker expects to perform.
                This is not a forced limit, but serves to calculate the `time_percentage` value passed into
                the Agent for time-dependent (decay) parameter calculations.
                If None, Worker will try to infer this value automatically.
        """
        super(Worker, self).__init__()
        self.num_environments = num_environments
        self.logger = logging.getLogger(__name__)

        self.update_rules = UpdateRules.from_spec(update_rules)

        # VectorEnv was passed in directly -> Use that one.
        if isinstance(env_spec, VectorEnv):
            self.vector_env = env_spec
            self.num_environments = self.vector_env.num_environments
            self.env_ids = ["env_{}".format(i) for i in range_(self.num_environments)]
        # `Env_spec` is for single envs inside a SequentialVectorEnv.
        elif env_spec is not None:
            self.vector_env = SequentialVectorEnv(env_spec=env_spec, num_environments=self.num_environments)
            self.env_ids = ["env_{}".format(i) for i in range_(self.num_environments)]
        # No env_spec.
        else:
            self.vector_env = None
            self.env_ids = []

        self.agent = agent
        self.frameskip = frameskip
        self.render = render

        self.episodes_since_update = 0

        self.max_timesteps = max_timesteps

        # Default val or None?
        #self.update_mode = "time_steps"

        self.worker_executes_exploration = worker_executes_exploration
        self.exploration_epsilon = exploration_epsilon

        self.episode_finish_callback = episode_finish_callback
        self.update_finish_callback = update_finish_callback

    def execute_timesteps(self, num_timesteps, max_timesteps_per_episode=0, update_rules=None,
                          use_exploration=True,
                          frameskip=1, reset=True):
        """
        Executes environment for a fixed number of timesteps.

        Args:
            num_timesteps (int): Number of time steps to execute.
            max_timesteps_per_episode (Optional[int]): Can be used to limit the number of timesteps per episode.
                Use None or 0 for no limit. Default: None.

            update_rules (Optional[dict]): The UpdateRules object to use. Overrides `self.update_rules` if provided.

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

    def execute_episodes(self, num_episodes, max_timesteps_per_episode=0, update_rules=None,
                         use_exploration=True, frameskip=None, reset=True):
        """
        Executes environment for a fixed number of episodes.

        Args:
            num_episodes (int): Number of episodes to execute.

            max_timesteps_per_episode (Optional[int]): Can be used to limit the number of timesteps per episode.
                Use None or 0 for no limit. Default: None.

            update_rules (Optional[dict]): The UpdateRules object to use. Overrides `self.update_rules` if provided.

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

    def update_if_necessary(self, update_rules, time_percentage):
        """
        Calls update on the agent according to the update schedule set for this worker.

        Args:
            update_rules (UpdateRules): The UpdateRules object to use.

            time_percentage (int): The percentage (0.0 to 1.0) of the already passed timesteps with respect to
                some max timestep value.

        Returns:
            float: The summed up loss (over all `update_rules.update_repeats` steps).
        """
        if update_rules.do_update:
            # Are we allowed to update?
            if self.agent.timesteps > update_rules.first_update_after_n_units and \
                    (self.agent.python_buffer_size == 0 or  # No update before some data in buffer
                     int(self.agent.timesteps / self.num_environments) >= self.agent.python_buffer_size):
                do_update = False
                # Updating according to one update mode:
                if update_rules.unit == "time_steps" and \
                        self.agent.timesteps % update_rules.update_every_n_units == 0:
                    do_update = True
                elif update_rules.unit == "episodes" and \
                        self.episodes_since_update == update_rules.update_every_n_units:
                    self.episodes_since_update = 0
                    do_update = True

                if do_update is True:
                    loss = 0
                    for _ in range_(update_rules.update_repeats):
                        ret = self.agent.update(time_percentage=time_percentage)
                        if isinstance(ret, dict):
                            loss += ret["loss"]
                        elif isinstance(ret, tuple):
                            loss += ret[0]
                        else:
                            loss += ret
                    return loss
        return None

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

    def log_finished_episode(self, episode_return, duration, timesteps, env_num=0):
        self.logger.debug("Finished episode: return={}, duration={}s, num-actions={}.".format(
            episode_return, duration, timesteps))

        if self.episode_finish_callback:
            self.episode_finish_callback(
                episode_return=episode_return, duration=duration, timesteps=timesteps, env_num=env_num
            )

    def log_finished_update(self, loss):
        self.logger.debug("Finished update: loss={}.".format(loss))

        if self.update_finish_callback:
            self.update_finish_callback(loss=loss)
