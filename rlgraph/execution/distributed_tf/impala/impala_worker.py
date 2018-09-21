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

import time

from rlgraph.agents.impala_agent import IMPALAAgent
from rlgraph.utils.util import default_dict
from rlgraph.execution.worker import Worker


class IMPALAWorker(Worker):

    def __init__(self, agent, **kwargs):
        """
        Args:
            agent (IMPALAAgent): The IMPALAAgent object to use.
            #num_steps (int): The number of steps (actions) to perform in the environment each rollout.
        """
        assert isinstance(agent, IMPALAAgent)

        frameskip = agent.environment_stepper.environment_spec.get("frameskip", 1)

        super(IMPALAWorker, self).__init__(agent=agent, frameskip=frameskip, **kwargs)

        self.logger.info(
            "Initialized IMPALA worker (type {}) with 1 environment '{}' running inside Agent's EnvStepper "
            "component.".format(self.agent.type, self.agent.environment_stepper.environment_spec)
        )

        # Global statistics.
        self.env_frames = 0
        self.finished_episode_rewards = list()
        self.finished_episode_durations = list()
        self.finished_episode_steps = list()

        # Accumulated return over the running episode.
        self.episode_returns = 0

        # The number of steps taken in the running episode.
        self.episode_timesteps = 0
        # Wall time of the last start of the running episode.
        #self.episode_starts = 0

    def execute_timesteps(self, num_timesteps, max_timesteps_per_episode=0, update_spec=None, use_exploration=True,
                          frameskip=None, reset=True):
        """
        Args:
            num_timesteps (Optional[int]): The maximum number of timesteps to run. At least one of `num_timesteps` or
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
        # Are we updating or just acting/observing?
        update_spec = default_dict(update_spec, self.agent.update_spec)
        self.set_update_schedule(update_spec)

        num_timesteps = num_timesteps or 0
        max_timesteps_per_episode = max_timesteps_per_episode or 0

        # Stats.
        timesteps_executed = 0
        episodes_executed = 0

        start = time.perf_counter()
        if reset is True:
            self.env_frames = 0
            #self.finished_episode_rewards = list()
            self.finished_episode_steps = list()

            #self.episode_returns = 0
            self.episode_timesteps = 0

            # TODO: Fix for vectorized Envs.
            self.agent.call_api_method("reset")

        # Only run everything for at most num_timesteps (if defined).
        while not (0 < num_timesteps <= timesteps_executed):
            # TODO right now everything comes back as single-env.
            out = self.agent.call_api_method(("perform_n_steps_and_insert_into_fifo", None, [0]))
            timesteps_executed += self.agent.worker_sample_size

            # Accumulate the reward over n env-steps (equals one action pick). n=self.frameskip.
            #rewards = out[2]
            terminals = out[3][1:]

            self.env_frames += self.frameskip * self.agent.worker_sample_size

            # Only render once per action.
            #if self.render:
            #    self.vector_env.environments[0].render()

            #for i in range_(self.num_environments):
            #    #self.episode_timesteps[i] += self.agent.worker_sample_size

            for j, terminal in enumerate(terminals):  # TODO: <- [i]
                self.episode_timesteps += 1

                if 0 < max_timesteps_per_episode <= self.episode_timesteps:
                    terminal = True

                if terminal:
                    episodes_executed += 1
                    self.finished_episode_steps.append(self.episode_timesteps)
                    self.logger.info("Finished episode: actions={}.".format(self.episode_timesteps))
                    self.episode_timesteps = 0

            num_timesteps_reached = (0 < num_timesteps <= timesteps_executed)

            if num_timesteps_reached:
                break

        total_time = (time.perf_counter() - start) or 1e-10

        # Return values for current episode(s) if None have been completed.
        #if len(self.finished_episode_rewards) == 0:
        #    #mean_episode_runtime = 0
        #    mean_episode_reward = np.mean(self.episode_returns)
        #    max_episode_reward = np.max(self.episode_returns)
        #    final_episode_reward = self.episode_returns[0]
        #else:
        #    #mean_episode_runtime = np.mean(self.finished_episode_durations)
        #    mean_episode_reward = np.mean(self.finished_episode_rewards)
        #    max_episode_reward = np.max(self.finished_episode_rewards)
        #    final_episode_reward = self.finished_episode_rewards[-1]

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
            #mean_episode_runtime=mean_episode_runtime,
            #mean_episode_reward=mean_episode_reward,
            #max_episode_reward=max_episode_reward,
            #final_episode_reward=final_episode_reward
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
        #self.logger.info("Mean episode runtime: {}s".format(results['mean_episode_runtime']))
        #self.logger.info("Mean episode reward: {}".format(results['mean_episode_reward']))
        #self.logger.info("Max. episode reward: {}".format(results['max_episode_reward']))
        #self.logger.info("Final episode reward: {}".format(results['final_episode_reward']))

        return results
