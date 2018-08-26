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
from six.moves import xrange as range_
import time

from rlgraph.utils.rlgraph_error import RLGraphError
from rlgraph.utils.util import default_dict
from rlgraph.execution.worker import Worker


class SingleThreadedWorker(Worker):

    def __init__(self, **kwargs):
        super(SingleThreadedWorker, self).__init__(**kwargs)

        self.logger.info("Initialized single-threaded executor with {} environments '{}' and Agent '{}'".format(
            self.num_environments, self.vector_env.get_env(), self.agent
        ))

        # Global statistics.
        self.env_frames = 0
        self.finished_episode_rewards = [list() for _ in range_(self.num_environments)]
        self.finished_episode_durations = [list() for _ in range_(self.num_environments)]
        self.finished_episode_timesteps = [list() for _ in range_(self.num_environments)]

        # Accumulated return over the running episode.
        self.episode_returns = [0 for _ in range_(self.num_environments)]

        # The number of steps taken in the running episode.
        self.episode_timesteps = [0 for _ in range_(self.num_environments)]
        # Whether the running episode has terminated.
        self.episode_terminals = [False for _ in range_(self.num_environments)]
        # Wall time of the last start of the running episode.
        self.episode_starts = [0 for _ in range_(self.num_environments)]
        # The current state of the running episode.
        self.env_states = [None for _ in range_(self.num_environments)]

    def execute_timesteps(self, num_timesteps, max_timesteps_per_episode=0, update_spec=None, use_exploration=True,
                          frameskip=None, reset=True):
        return self._execute(
            num_timesteps=num_timesteps,
            max_timesteps_per_episode=max_timesteps_per_episode,
            use_exploration=use_exploration,
            update_spec=update_spec,
            frameskip=frameskip,
            reset=reset
        )

    def execute_episodes(self, num_episodes, max_timesteps_per_episode=0, update_spec=None, use_exploration=True,
                         frameskip=None, reset=True):
        return self._execute(
            num_episodes=num_episodes,
            max_timesteps_per_episode = max_timesteps_per_episode,
            use_exploration=use_exploration,
            update_spec=update_spec,
            frameskip=frameskip,
            reset=reset
        )

    def _execute(
        self,
        num_timesteps=None,
        num_episodes=None,
        max_timesteps_per_episode=None,
        use_exploration=True,
        update_spec=None,
        frameskip=None,
        reset=True
    ):
        """
        Actual implementation underlying `execute_timesteps` and `execute_episodes`.

        Args:
            num_timesteps (Optional[int]): The maximum number of timesteps to run. At least one of `num_timesteps` or
                `num_episodes` must be provided.
            num_episodes (Optional[int]): The maximum number of episodes to run. At least one of `num_timesteps` or
                `num_episodes` must be provided.
            use_exploration (Optional[bool]): Indicates whether to utilize exploration (epsilon or noise based)
                when picking actions. Default: True.
            max_timesteps_per_episode (Optional[int]): Can be used to limit the number of timesteps per episode.
                Use None or 0 for no limit. Default: None.
            update_spec (Optional[dict]): Update parameters. If None, the worker only performs rollouts.
                Matches the structure of an Agent's update_spec dict and will be "defaulted" by that dict.
                See `input_parsing/parse_update_spec.py` for more details.
            frameskip (Optional[int]): How often actions are repeated after retrieving them from the agent.
                Rewards are accumulated over the number of skips. Use None for the Worker's default value.
            reset (bool): Whether to reset the environment and all the Worker's internal counters.
                Default: True.

        Returns:
            dict: Execution statistics.
        """
        assert num_timesteps is not None or num_episodes is not None,\
            "ERROR: One of `num_timesteps` or `num_episodes` must be provided!"
        # Are we updating or just acting/observing?
        update_spec = default_dict(update_spec, self.agent.update_spec)
        self.set_update_schedule(update_spec)

        num_timesteps = num_timesteps or 0
        num_episodes = num_episodes or 0
        max_timesteps_per_episode = [max_timesteps_per_episode or 0 for _ in range_(self.num_environments)]
        frameskip = frameskip or self.frameskip

        # Stats.
        timesteps_executed = 0
        episodes_executed = 0

        start = time.monotonic()
        if reset is True:
            self.env_frames = 0
            self.finished_episode_rewards = [list() for _ in range_(self.num_environments)]
            self.finished_episode_durations = [list() for _ in range_(self.num_environments)]
            self.finished_episode_timesteps = [list() for _ in range_(self.num_environments)]

            for i in range_(self.num_environments):
                self.episode_returns[i] = 0
                self.episode_timesteps[i] = 0
                self.episode_terminals[i] = False
                self.episode_starts[i] = time.monotonic()
            self.env_states = self.vector_env.reset()
            self.agent.reset()
        elif self.env_states[0] is None:
            raise RLGraphError("Runner must be reset at the very beginning. Environment is in invalid state.")

        # Only run everything for at most num_timesteps (if defined).
        while not (0 < num_timesteps <= timesteps_executed):

            if self.render:
                # This renders the first underlying environment.
                self.vector_env.render()

            self.env_states = self.agent.state_space.force_batch(self.env_states)

            actions, preprocessed_states = self.agent.get_action(
                states=self.env_states, use_exploration=use_exploration, extra_returns="preprocessed_states"
            )

            # Accumulate the reward over n env-steps (equals one action pick). n=self.frameskip.
            env_rewards = [0 for _ in range_(self.num_environments)]
            next_states = None
            for _ in range_(frameskip):
                next_states, step_rewards, self.episode_terminals, infos = self.vector_env.step(actions=actions)

                self.env_frames += self.num_environments
                for i, step_reward in enumerate(step_rewards):
                    env_rewards[i] += step_reward
                if np.any(self.episode_terminals):
                    break

            # Only render once per action.
            if self.render:
                self.vector_env.environments[0].render()

            for i in range_(self.num_environments):
                self.episode_returns[i] += env_rewards[i]
                self.episode_timesteps[i] += 1

                if 0 < max_timesteps_per_episode[i] <= self.episode_timesteps[i]:
                    self.episode_terminals[i] = True

                # Do accounting for finished episodes.
                if self.episode_terminals[i]:
                    episodes_executed += 1
                    self.finished_episode_rewards[i].append(self.episode_returns[i])
                    self.finished_episode_durations[i].append(time.monotonic() - self.episode_starts[i])
                    self.finished_episode_timesteps[i].append(self.episode_timesteps[i])
                    self.logger.info("Finished episode: reward={}, actions={}, duration={}s.".format(
                        self.episode_returns[i], self.episode_timesteps[i], self.finished_episode_durations[i][-1]))

                    # Reset this environment and its preprocecssor stack.
                    self.env_states[i] = self.vector_env.reset(i)
                    self.episode_returns[i] = 0
                    self.episode_timesteps[i] = 0
                    self.episode_starts[i] = time.monotonic()
                else:
                    # Otherwise assign states to next states
                    self.env_states[i] = next_states[i]

                # Observe per environment.
                self.agent.observe(
                    preprocessed_states=preprocessed_states[i], actions=actions[i], internals=[],
                    rewards=env_rewards[i], terminals=self.episode_terminals[i], env_id=self.env_ids[i]
                )
            self.update_if_necessary()
            timesteps_executed += self.num_environments
            num_timesteps_reached = (0 < num_timesteps <= timesteps_executed)

            if 0 < num_episodes <= episodes_executed or num_timesteps_reached:
                break

        total_time = (time.monotonic() - start) or 1e-10

        # Return values for current episode(s) if None have been completed.
        if len(self.finished_episode_rewards) == 0:
            mean_episode_runtime = 0
            mean_episode_reward = np.mean(self.episode_returns)
            max_episode_reward = np.max(self.episode_returns)
            final_episode_reward = self.episode_returns[0]
        else:
            all_finished_durations = list()
            all_finished_rewards = list()
            for i in range_(self.num_environments):
                all_finished_rewards.extend(self.finished_episode_rewards[i])
                all_finished_durations.extend(self.finished_episode_durations[i])
            mean_episode_runtime = np.mean(all_finished_durations)
            mean_episode_reward = np.mean(all_finished_rewards)
            max_episode_reward = np.max(all_finished_rewards)
            final_episode_reward = all_finished_rewards[-1]

        results = dict(
            runtime=total_time,
            # Agent act/observe throughput.
            timesteps_executed=timesteps_executed,
            ops_per_second=(timesteps_executed / total_time),
            # Env frames including action repeats.
            env_frames=self.env_frames,
            env_frames_per_second=(self.env_frames / total_time),
            episodes_executed=episodes_executed,
            episodes_per_minute=(episodes_executed/(total_time / 60)),
            mean_episode_runtime=mean_episode_runtime,
            mean_episode_reward=mean_episode_reward,
            max_episode_reward=max_episode_reward,
            final_episode_reward=final_episode_reward
        )

        # Total time of run.
        self.logger.info("Finished execution in {} s".format(total_time))
        # Total (RL) timesteps (actions) done (and timesteps/sec).
        self.logger.info("Time steps (actions) executed: {} ({} ops/s)".
                         format(results['timesteps_executed'], results['ops_per_second']))
        # Total env-timesteps done (including action repeats) (and env-timesteps/sec).
        self.logger.info("Env frames executed (incl. action repeats): {} ({} frames/s)".
                         format(results['env_frames'], results['env_frames_per_second']))
        # Total episodes done (and episodes/min).
        self.logger.info("Episodes finished: {} ({} episodes/min)".
                         format(results['episodes_executed'], results['episodes_per_minute']))
        self.logger.info("Mean episode runtime: {}s".format(results['mean_episode_runtime']))
        self.logger.info("Mean episode reward: {}".format(results['mean_episode_reward']))
        self.logger.info("Max. episode reward: {}".format(results['max_episode_reward']))
        self.logger.info("Final episode reward: {}".format(results['final_episode_reward']))

        return results
