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

from copy import deepcopy

import numpy as np
from rlgraph.components import PreprocessorStack
from six.moves import xrange as range_
import time

from rlgraph.utils.rlgraph_errors import RLGraphError
from rlgraph.utils.util import default_dict
from rlgraph.execution.worker import Worker


class SingleThreadedWorker(Worker):

    def __init__(self, preprocessing_spec=None, worker_executes_preprocessing=True, **kwargs):
        super(SingleThreadedWorker, self).__init__(**kwargs)

        self.logger.info("Initialized single-threaded executor with {} environments '{}' and Agent '{}'".format(
            self.num_environments, self.vector_env.get_env(), self.agent
        ))
        self.worker_executes_preprocessing = worker_executes_preprocessing
        if self.worker_executes_preprocessing:
            assert preprocessing_spec is not None
            self.preprocessors = {}
            self.state_is_preprocessed = {}
            for env_id in self.env_ids:
                self.preprocessors[env_id] = self.setup_preprocessor(
                    preprocessing_spec, self.vector_env.state_space.with_batch_rank()
                )
                self.state_is_preprocessed[env_id] = False

        self.apply_preprocessing = not self.worker_executes_preprocessing
        self.preprocessed_states_buffer = np.zeros(
            shape=(self.num_environments,) + self.agent.preprocessed_state_space.shape,
            dtype=self.agent.preprocessed_state_space.dtype
        )

        # Global statistics.
        self.env_frames = 0
        self.finished_episode_rewards = [[] for _ in range_(self.num_environments)]
        self.finished_episode_durations = [[] for _ in range_(self.num_environments)]
        self.finished_episode_timesteps = [[] for _ in range_(self.num_environments)]

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

    def setup_preprocessor(self, preprocessing_spec, in_space):
        if preprocessing_spec is not None:
            # TODO move ingraph for python component assembly.
            preprocessing_spec = deepcopy(preprocessing_spec)
            in_space = deepcopy(in_space)
            # Set scopes.
            scopes = [preprocessor["scope"] for preprocessor in preprocessing_spec]
            # Set backend to python.
            for spec in preprocessing_spec:
                spec["backend"] = "python"
            processor_stack = PreprocessorStack(*preprocessing_spec, backend="python")
            build_space = in_space
            for sub_comp_scope in scopes:
                processor_stack.sub_components[sub_comp_scope].create_variables(input_spaces=dict(
                    preprocessing_inputs=build_space
                ), action_space=None)
                build_space = processor_stack.sub_components[sub_comp_scope].get_preprocessed_space(build_space)
            processor_stack.reset()
            return processor_stack
        else:
            return None

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

        start = time.perf_counter()
        episode_terminals = self.episode_terminals
        if reset is True:
            self.env_frames = 0
            self.finished_episode_rewards = [[] for _ in range_(self.num_environments)]
            self.finished_episode_durations = [[] for _ in range_(self.num_environments)]
            self.finished_episode_timesteps = [[] for _ in range_(self.num_environments)]

            for i, env_id in enumerate(self.env_ids):
                self.episode_returns[i] = 0
                self.episode_timesteps[i] = 0
                self.episode_terminals[i] = False
                self.episode_starts[i] = time.perf_counter()
                if self.worker_executes_preprocessing:
                    self.state_is_preprocessed[env_id] = False

            self.env_states = self.vector_env.reset_all()
            self.agent.reset()
        elif self.env_states[0] is None:
            raise RLGraphError("Runner must be reset at the very beginning. Environment is in invalid state.")

        # Only run everything for at most num_timesteps (if defined).
        env_states = self.env_states
        while not (0 < num_timesteps <= timesteps_executed):
            if self.render:
                # This renders the first underlying environment.
                self.vector_env.render()

            if self.worker_executes_preprocessing:
                for i, env_id in enumerate(self.env_ids):
                    state = self.agent.state_space.force_batch(env_states[i])
                    if self.preprocessors[env_id] is not None:
                        if self.state_is_preprocessed[env_id] is False:
                            self.preprocessed_states_buffer[i] = self.preprocessors[env_id].preprocess(state)
                            self.state_is_preprocessed[env_id] = True
                    else:
                        self.preprocessed_states_buffer[i] = env_states[i]
                # TODO extra returns when worker is not applying preprocessing.
                actions = self.agent.get_action(
                    states=self.preprocessed_states_buffer, use_exploration=use_exploration,
                    apply_preprocessing=self.apply_preprocessing
                )
                preprocessed_states = np.array(self.preprocessed_states_buffer)
            else:
                actions, preprocessed_states = self.agent.get_action(
                    states=np.array(env_states), use_exploration=use_exploration,
                    apply_preprocessing=True, extra_returns="preprocessed_states"
                )

            # Accumulate the reward over n env-steps (equals one action pick). n=self.frameskip.
            env_rewards = [0 for _ in range_(self.num_environments)]
            next_states = None
            for _ in range_(frameskip):
                next_states, step_rewards, episode_terminals, infos = self.vector_env.step(actions=actions)

                self.env_frames += self.num_environments
                for i, step_reward in enumerate(step_rewards):
                    env_rewards[i] += step_reward
                if np.any(episode_terminals):
                    break

            # Only render once per action.
            if self.render:
                self.vector_env.environments[0].render()

            for i, env_id in enumerate(self.env_ids):
                self.episode_returns[i] += env_rewards[i]
                self.episode_timesteps[i] += 1

                if 0 < max_timesteps_per_episode[i] <= self.episode_timesteps[i]:
                    episode_terminals[i] = True
                if self.worker_executes_preprocessing:
                    self.state_is_preprocessed[env_id] = False
                # Do accounting for finished episodes.
                if episode_terminals[i]:
                    episodes_executed += 1
                    self.finished_episode_rewards[i].append(self.episode_returns[i])
                    self.finished_episode_durations[i].append(time.perf_counter() - self.episode_starts[i])
                    self.finished_episode_timesteps[i].append(self.episode_timesteps[i])
                    self.logger.debug("Finished episode: reward={}, actions={}, duration={}s.".format(
                        self.episode_returns[i], self.episode_timesteps[i], self.finished_episode_durations[i][-1]))

                    # Reset this environment and its preprocecssor stack.
                    env_states[i] = self.vector_env.reset(i)
                    if self.worker_executes_preprocessing and self.preprocessors[env_id] is not None:
                        self.preprocessors[env_id].reset()
                        # This re-fills the sequence with the reset state.
                        state = self.agent.state_space.force_batch(env_states[i])
                        # Pre - process, add to buffer
                        self.preprocessed_states_buffer[i] = np.array(self.preprocessors[env_id].preprocess(state))
                        self.state_is_preprocessed[env_id] = True

                    self.episode_returns[i] = 0
                    self.episode_timesteps[i] = 0
                    self.episode_starts[i] = time.perf_counter()
                else:
                    # Otherwise assign states to next states
                    env_states[i] = next_states[i]

                if self.worker_executes_preprocessing and self.preprocessors[env_id] is not None:
                    next_state = self.agent.state_space.force_batch(env_states[i])
                    next_states[i] = np.array(self.preprocessors[env_id].preprocess(next_state))
                # TODO: If worker does not execute preprocessing, next state is not preprocessed here.
                # Observe per environment.
                self.agent.observe(
                    preprocessed_states=preprocessed_states[i], actions=actions[i], internals=[],
                    rewards=env_rewards[i], next_states=next_states[i],
                    terminals=episode_terminals[i], env_id=self.env_ids[i]
                )
            self.update_if_necessary()
            timesteps_executed += self.num_environments
            num_timesteps_reached = (0 < num_timesteps <= timesteps_executed)

            if 0 < num_episodes <= episodes_executed or num_timesteps_reached:
                break

        total_time = (time.perf_counter() - start) or 1e-10

        # Return values for current episode(s) if None have been completed.
        if episodes_executed == 0:
            mean_episode_runtime = 0
            mean_episode_reward = np.mean(self.episode_returns)
            max_episode_reward = np.max(self.episode_returns)
            final_episode_reward = self.episode_returns[0]
        else:
            all_finished_durations = []
            all_finished_rewards = []
            for i in range_(self.num_environments):
                all_finished_rewards.extend(self.finished_episode_rewards[i])
                all_finished_durations.extend(self.finished_episode_durations[i])
            mean_episode_runtime = np.mean(all_finished_durations)
            mean_episode_reward = np.mean(all_finished_rewards)
            max_episode_reward = np.max(all_finished_rewards)
            final_episode_reward = all_finished_rewards[-1]

        self.episode_terminals = episode_terminals
        self.env_states = env_states
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
