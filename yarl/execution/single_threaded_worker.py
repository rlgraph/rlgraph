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

from yarl.execution.worker import Worker
from six.moves import xrange
import time
import numpy as np


class SingleThreadedWorker(Worker):

    def __init__(self, **kwargs):
        super(SingleThreadedWorker, self).__init__(**kwargs)

        self.logger.info("Initialized single-threaded executor with\n environment id {} and agent {}".format(
            self.environment, self.agent
        ))

    def execute_timesteps(self, num_timesteps, deterministic=False):
        executed = 0
        episode_rewards = []
        episode_durations = []
        episode_steps = []
        start = time.monotonic()

        while executed < num_timesteps:
            episode_reward = 0
            episode_step = 0
            terminal = False
            episode_start = time.monotonic()
            state = self.environment.reset()

            while True:
                reward = 0
                actions = self.agent.get_action(states=state, deterministic=deterministic)

                for repeat in xrange(self.repeat_actions):
                    state, terminal, step_reward, info = self.environment.step(actions=actions)
                    reward += step_reward
                    if terminal or executed >= num_timesteps:
                        break

                self.agent.observe(states=state, actions=actions, internals=None, reward=reward, terminal=terminal)
                episode_reward += reward
                executed += 1
                episode_step += 1
                if terminal or executed >= num_timesteps:
                    break

            episode_rewards.append(episode_reward)
            episode_durations.append(time.monotonic() - episode_start)
            episode_steps.append(episode_step)
            self.logger.info("Finished episode, reward: {}, steps {}, duration {}.".format(
                episode_reward, episode_step, episode_durations[-1]))

        end = time.monotonic() - start
        self.logger.info("Finished executing {} time steps in {} s".format(num_timesteps, end))
        # TODO check how we define frames regarding action repeat.
        self.logger.info("Throughput: {} ops/s".format(num_timesteps / end))
        self.logger.info("Mean episode reward: {}".format(np.mean(episode_rewards)))
        self.logger.info("Final episode reward: {}".format(episode_rewards[-1]))

        return dict(
            runtime=end,
            ops_per_second=(num_timesteps / end),
            episodes_executed=len(episode_durations),
            mean_episode_runtime=np.mean(episode_durations),
            mean_episode_reward=np.mean(episode_rewards),
            max_episode_reward=np.max(episode_rewards),
            final_episode_reward=episode_rewards[-1]
        )

    def execute_episodes(self, num_episodes, max_timesteps_per_episode, deterministic=False):
        executed = 0
        episodes_executed = 0
        episode_rewards = []
        episode_durations = []
        episode_steps = []
        start = time.monotonic()

        while episodes_executed < num_episodes:
            episode_reward = 0
            episode_step = 0
            terminal = False
            episode_start= time.monotonic()
            state = self.environment.reset()

            while True:
                reward = 0
                actions = self.agent.get_action(states=state, deterministic=deterministic)

                for repeat in xrange(self.repeat_actions):
                    state, terminal, step_reward, info = self.environment.step(actions=actions)
                    reward += step_reward
                    if terminal or episodes_executed >= num_episodes:
                        break

                self.agent.observe(states=state, actions=actions, internals=None, reward=reward, terminal=terminal)
                episode_reward += reward
                executed += 1
                episode_step += 1
                if terminal or episodes_executed >= num_episodes:
                    break

            episodes_executed += 1
            episode_rewards.append(episode_reward)
            episode_durations.append(time.monotonic() - episode_start)
            episode_steps.append(episode_step)
            self.logger.info("Finished episode, reward: {}, steps {}, duration {}.".format(
                episode_reward, episode_step, episode_durations[-1]))

        end = time.monotonic() - start
        self.logger.info("Finished executing {} episodes in {} s".format(num_episodes, end))
        # TODO check how we define frames regarding action repeat.
        self.logger.info("Throughput: {} ops/s".format(executed / end))
        self.logger.info("Mean episode reward: {}".format(np.mean(episode_rewards)))
        self.logger.info("Final episode reward: {}".format(episode_rewards[-1]))

        return dict(
            runtime=end,
            ops_per_second=(executed / end),
            timesteps_executed=executed,
            mean_episode_runtime=np.mean(episode_durations),
            mean_episode_reward=np.mean(episode_rewards),
            max_episode_reward=np.max(episode_rewards),
            final_episode_reward=episode_rewards[-1]
        )
