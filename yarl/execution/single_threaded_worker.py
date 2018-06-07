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

import numpy as np
from six.moves import xrange
import time

from yarl.execution.worker import Worker


class SingleThreadedWorker(Worker):

    def __init__(self, **kwargs):
        super(SingleThreadedWorker, self).__init__(**kwargs)

        self.logger.info("Initialized single-threaded executor with\n environment id {} and agent {}".format(
            self.environment, self.agent
        ))

    def execute_timesteps(self, num_timesteps, max_timesteps_per_episode=0, update_schedule=None, deterministic=False):
        return self._execute(num_timesteps=num_timesteps, max_timesteps_per_episode=max_timesteps_per_episode,
                             deterministic=deterministic)

    def execute_episodes(self, num_episodes, max_timesteps_per_episode=0, update_schedule=None, deterministic=False):
        return self._execute(num_episodes=num_episodes, max_timesteps_per_episode=max_timesteps_per_episode,
                             deterministic=deterministic)

    def _execute(
        self,
        num_timesteps=None,
        num_episodes=None,
        deterministic=False,
        max_timesteps_per_episode=None,
        update_schedule=None
    ):
        """
        Actual implementation underlying `execute_timesteps` and `execute_episodes`.

        Args:
            num_timesteps (Optional[int]): The maximum number of timesteps to run. At least one of `num_timesteps` or
                `num_episodes` must be provided.
            num_episodes (Optional[int]): The maximum number of episodes to run. At least one of `num_timesteps` or
                `num_episodes` must be provided.
            deterministic (bool): Whether to execute actions deterministically or not.
                Default: False.
            max_timesteps_per_episode (Optional[int]): Can be used to limit the number of timesteps per episode.
                Use None or 0 for no limit. Default: None.
            update_schedule (Optional[dict]): Update parameters. If None, the worker only peforms rollouts.
                Expects keys 'update_interval' to indicate how frequent update is called, 'num_updates'
                to indicate how many updates to perform every update interval, and 'steps_before_update' to indicate
                how many steps to perform before beginning to update.
        Returns:
            dict: Execution statistics.
        """
        assert num_timesteps is not None or num_episodes is not None, "ERROR: One of `num_timesteps` or `num_episodes` " \
                                                                      "must be provided!"
        # Are we updating
        updating = False
        update_interval = None
        update_steps = None
        steps_before_update = None
        if update_schedule is not None:
            updating = True
            steps_before_update = update_schedule['steps_before_update']
            update_interval = update_schedule['update_interval']
            update_steps = update_schedule['update_steps']

        num_timesteps = num_timesteps or 0
        num_episodes = num_episodes or 0
        max_timesteps_per_episode = max_timesteps_per_episode or 0

        # Stats.
        timesteps_executed = 0
        episodes_executed = 0
        env_frames = 0
        episode_rewards = []
        episode_durations = []
        episode_steps = []
        start = time.monotonic()

        # Only run everything for at most num_timesteps (if defined).
        while not (num_timesteps > 0 and timesteps_executed < num_timesteps):
            # The reward accumulated over one episode.
            episode_reward = 0
            # The number of steps taken in the episode.
            episode_timestep = 0
            # Whether the episode has terminated.
            terminal = False

            # Start a new episode.
            episode_start = time.monotonic()  # wall time
            state = self.environment.reset()
            while True:
                actions = self.agent.get_action(states=state, deterministic=deterministic)

                # Accumulate the reward over n env-steps (equals one action pick). n=self.repeat_actions
                reward = 0
                for _ in xrange(self.repeat_actions):
                    #
                    state, step_reward, terminal, info = self.environment.step(actions=actions)
                    env_frames += 1
                    reward += step_reward
                    if terminal:
                        break

                self.agent.observe(states=state, actions=actions, internals=None, reward=reward, terminal=terminal)

                if updating:
                    if timesteps_executed > steps_before_update and timesteps_executed % update_interval == 0:
                        for _ in xrange(update_steps):
                            self.agent.update()
                episode_reward += reward
                timesteps_executed += 1
                episode_timestep += 1
                # Is the episode finished or do we have to terminate it prematurely because of other restrictions?
                if terminal or (0 < num_timesteps <= timesteps_executed) or \
                        (0 < max_timesteps_per_episode <= episode_timestep):
                    break

            episodes_executed += 1
            episode_rewards.append(episode_reward)
            episode_durations.append(time.monotonic() - episode_start)
            episode_steps.append(episode_timestep)
            self.logger.info("Finished episode: reward={}, actions={}, duration={}s.".format(
                episode_reward, episode_timestep, episode_durations[-1]))

            if 0 < num_episodes <= episodes_executed:
                break

        total_time = (time.monotonic() - start) or 1e-10
        # Total time of run.
        self.logger.info("Finished execution in {} s".format(total_time))
        # Total (RL) timesteps (actions) done (and timesteps/sec).
        self.logger.info("Time steps (actions) executed: {} ({} ops/s)".
                         format(timesteps_executed, timesteps_executed / total_time))
        # Total env-timesteps done (including action repeats) (and env-timesteps/sec).
        self.logger.info("Env frames executed (incl. action repeats): {} ({} frames/s)".
                         format(env_frames, env_frames / total_time))
        # Total episodes done (and episodes/min).
        self.logger.info("Episodes executed: {} ({} episodes/min)".
                         format(episodes_executed, episodes_executed/(total_time*60)))
        # Mean episode runtime.
        self.logger.info("Mean episode runtime: {}s".format(np.mean(episode_durations)))
        # Mean episode reward.
        self.logger.info("Mean episode reward: {}".format(np.mean(episode_rewards)))
        # Max. episode reward.
        self.logger.info("Max. episode reward: {}".format(np.max(episode_rewards)))
        # Last episode reward.
        self.logger.info("Final episode reward: {}".format(episode_rewards[-1]))

        return dict(
            runtime=total_time,
            # Agent act/observe throughput.
            timesteps_executed=timesteps_executed,
            ops_per_second=(timesteps_executed / total_time),
            # Env frames including action repeats.
            env_frames=env_frames,
            env_frames_per_second=(env_frames / total_time),
            episodes_executed=episodes_executed,
            episodes_per_minute=(episodes_executed/(total_time*60)),
            mean_episode_runtime=np.mean(episode_durations),
            mean_episode_reward=np.mean(episode_rewards),
            max_episode_reward=np.max(episode_rewards),
            final_episode_reward=episode_rewards[-1]
        )

