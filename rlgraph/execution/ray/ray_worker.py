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
from six.moves import xrange as range_
import time

from rlgraph import get_distributed_backend
from rlgraph.utils.util import SMALL_NUMBER
from rlgraph.components.neural_networks.preprocessor_stack import PreprocessorStack
from rlgraph.environments.sequential_vector_env import SequentialVectorEnv
from rlgraph.execution.environment_sample import EnvironmentSample
from rlgraph.execution.ray import RayExecutor
from rlgraph.execution.ray.ray_actor import RayActor
from rlgraph.execution.ray.ray_util import ray_compress

if get_distributed_backend() == "ray":
    import ray


class RayWorker(RayActor):
    """
    Ray wrapper for single threaded worker, provides further api methods to interact
    with the agent used in the worker.
    """

    def __init__(self, agent_config, worker_spec, env_spec, frameskip=1, auto_build=False):
        """
        Creates agent and environment for Ray worker.

        Args:
            agent_config (dict): Agent configuration dict.
            worker_spec (dict): Worker parameters.
            env_spec (dict): Environment config for environment to run.
            frameskip (int): How often actions are repeated after retrieving them from the agent.
        """
        assert get_distributed_backend() == "ray"
        # Internal frameskip of env.
        self.env_frame_skip = env_spec.get("frameskip", 1)
        # Worker computes weights for prioritized sampling.
        worker_spec = deepcopy(worker_spec)
        self.worker_sample_size = worker_spec.pop("worker_sample_size")
        self.worker_computes_weights = worker_spec.pop("worker_computes_weights", True)
        self.n_step_adjustment = worker_spec.pop("n_step_adjustment", 1)
        self.num_environments = worker_spec.pop("num_worker_environments", 1)
        self.env_ids = ["env_{}".format(i) for i in range_(self.num_environments)]
        self.auto_build = auto_build
        num_background_envs = worker_spec.pop("num_background_envs", 1)

        # TODO from spec once we decided on generic vectorization.
        self.vector_env = SequentialVectorEnv(self.num_environments, env_spec, num_background_envs)

        # Then update agent config.
        agent_config['state_space'] = self.vector_env.state_space
        agent_config['action_space'] = self.vector_env.action_space

        ray_exploration = worker_spec.pop("ray_exploration", None)
        self.worker_executes_exploration = worker_spec.pop("worker_executes_exploration", False)
        self.ray_exploration_set = False
        if ray_exploration is not None:
            # Update worker with worker specific constant exploration value.
            # TODO too many levels?
            assert agent_config["exploration_spec"]["epsilon_spec"]["decay_spec"]["type"] == "constant_decay", \
                "ERROR: If using Ray's constant exploration, exploration type must be 'constant_decay'."
            if self.worker_executes_exploration:
                agent_config["exploration_spec"] = None
                self.exploration_epsilon = ray_exploration
            else:
                agent_config["exploration_spec"]["epsilon_spec"]["decay_spec"]["constant_value"] = ray_exploration
                self.ray_exploration_set = True

        self.discount = agent_config.get("discount", 0.99)
        # Python based preprocessor as image resizing is broken in TF.

        self.preprocessors = {}
        preprocessing_spec = agent_config.get("preprocessing_spec", None)
        self.is_preprocessed = {}
        for env_id in self.env_ids:
            self.preprocessors[env_id] = self.setup_preprocessor(
                preprocessing_spec, self.vector_env.state_space.with_batch_rank()
            )
            self.is_preprocessed[env_id] = False
        self.agent = self.setup_agent(agent_config, worker_spec)
        self.worker_frameskip = frameskip

        # Save these so they can be fetched after training if desired.
        self.finished_episode_rewards = [[] for _ in range_(self.num_environments)]
        self.finished_episode_timesteps = [[] for _ in range_(self.num_environments)]
        # Total times sample the "real" wallclock time from start to end for each episode.
        self.finished_episode_total_times = [[] for _ in range_(self.num_environments)]
        # Sample times stop the wallclock time counter between runs, so only the sampling time is accounted for.
        self.finished_episode_sample_times = [[] for _ in range_(self.num_environments)]

        self.total_worker_steps = 0
        self.episodes_executed = 0

        # Step time and steps done per call to execute_and_get to measure throughput of this worker.
        self.sample_times = []
        self.sample_steps = []
        self.sample_env_frames = []

        # To continue running through multiple exec calls.
        self.last_states = self.vector_env.reset_all()
        self.agent.reset()

        self.zero_batched_state = np.zeros((1,) + self.agent.preprocessed_state_space.shape)
        self.zero_unbatched_state = np.zeros(self.agent.preprocessed_state_space.shape)
        self.preprocessed_states_buffer = np.zeros(
            shape=(self.num_environments,) + self.agent.preprocessed_state_space.shape,
            dtype=self.agent.preprocessed_state_space.dtype
        )
        self.last_ep_timesteps = [0 for _ in range_(self.num_environments)]
        self.last_ep_rewards = [0 for _ in range_(self.num_environments)]
        self.last_ep_start_timestamps = [0.0 for _ in range_(self.num_environments)]
        self.last_ep_start_initialized = False  # initialize on first `execute_and_get_timesteps()` call
        self.last_ep_sample_times = [0.0 for _ in range_(self.num_environments)]

        # Was the last state a terminal state so env should be reset in next call?
        self.last_terminals = [False for _ in range_(self.num_environments)]

    def get_constructor_success(self):
        """
        For debugging: fetch the last attribute. Will fail if constructor failed.
        """
        return not self.last_terminals[0]

    @classmethod
    def as_remote(cls, num_cpus=None, num_gpus=None):
        return ray.remote(num_cpus=num_cpus, num_gpus=num_gpus)(cls)

    def init_agent(self):
        """
        Builds the agent. This is done as a separate task because meta graph
        generation can take long.
        """
        self.agent.build()
        return True

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

    def setup_agent(self, agent_config, worker_spec):
        """
        Sets up agent, potentially modifying its configuration via worker specific settings.
        """
        sample_exploration = worker_spec.pop("sample_exploration", False)
        # Adjust exploration for this worker.
        if sample_exploration:
            assert self.ray_exploration_set is False, "ERROR: Cannot sample exploration if ray exploration is used."
            exploration_min_value = worker_spec.pop("exploration_min_value", 0.0)
            epsilon_spec = agent_config["exploration_spec"]["epsilon_spec"]

            if epsilon_spec is not None and "decay_spec" in epsilon_spec:
                decay_from = epsilon_spec["decay_spec"]["from"]
                assert decay_from >= exploration_min_value, \
                    "Min value for exploration sampling must be smaller than" \
                    "decay_from {} in exploration_spec but is {}.".format(decay_from, exploration_min_value)

                # Sample a new initial epsilon from the interval [exploration_min_value, decay_from).
                sampled_from = np.random.uniform(low=exploration_min_value, high=decay_from)
                epsilon_spec["decay_spec"]["from"] = sampled_from

        # Worker execution spec may differ from controller/learner.
        worker_exec_spec = worker_spec.get("execution_spec", None)
        if worker_exec_spec is not None:
            agent_config.update(execution_spec=worker_exec_spec)

        # Build lazily per default.
        agent_config.update(auto_build=self.auto_build)
        return RayExecutor.build_agent_from_config(agent_config)

    def execute_and_get_timesteps(
        self,
        num_timesteps,
        max_timesteps_per_episode=0,
        use_exploration=True,
        break_on_terminal=False
    ):
        """
        Collects and returns time step experience.

        Args:
            break_on_terminal (Optional[bool]): If true, breaks when a terminal is encountered. If false,
                executes exactly 'num_timesteps' steps.
        """
        # Initialize start timestamps. Initializing the timestamps here should make the observed execution timestamps
        # more accurate, as there might be delays between the worker initialization and actual sampling start.
        if not self.last_ep_start_initialized:
            for i, timestamp in enumerate(self.last_ep_start_timestamps):
                self.last_ep_start_timestamps[i] = time.perf_counter()
            self.last_ep_start_initialized = True

        start = time.monotonic()
        timesteps_executed = 0
        episodes_executed = [0 for _ in range_(self.num_environments)]
        env_frames = 0

        # Final result batch.
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_terminals = [], [], [], [], []

        # Running trajectories.
        sample_states, sample_actions, sample_rewards, sample_terminals = {}, {}, {}, {}
        next_states = [np.zeros_like(self.last_states) for _ in range_(self.num_environments)]

        # Reset envs and Agent either if finished an episode in current loop or if last state
        # from previous execution was terminal for that environment.
        for i, env_id in enumerate(self.env_ids):
            sample_states[env_id] = []
            sample_actions[env_id] = []
            sample_rewards[env_id] = []
            sample_terminals[env_id] = []

        env_states = self.last_states
        current_episode_rewards = self.last_ep_rewards
        current_episode_timesteps = self.last_ep_timesteps
        current_episode_start_timestamps = self.last_ep_start_timestamps
        current_episode_sample_times = self.last_ep_sample_times

        # Whether the episode in each env has terminated.
        terminals = [False for _ in range_(self.num_environments)]
        while timesteps_executed < num_timesteps:
            current_iteration_start_timestamp = time.perf_counter()
            for i, env_id in enumerate(self.env_ids):
                state = self.agent.state_space.force_batch(env_states[i])
                if self.preprocessors[env_id] is not None:
                    if self.is_preprocessed[env_id] is False:
                        self.preprocessed_states_buffer[i] = self.preprocessors[env_id].preprocess(state)
                        self.is_preprocessed[env_id] = True
                else:
                    self.preprocessed_states_buffer[i] = env_states[i]

            actions = self.get_action(states=self.preprocessed_states_buffer,
                                      use_exploration=use_exploration, apply_preprocessing=False)
            next_states, step_rewards, terminals, infos = self.vector_env.step(actions=actions)
            # Worker frameskip not needed as done in env.
            # for _ in range_(self.worker_frameskip):
            #     next_states, step_rewards, terminals, infos = self.vector_env.step(actions=actions)
            #     env_frames += self.num_environments
            #
            #     for i, env_id in enumerate(self.env_ids):
            #         rewards[env_id] += step_rewards[i]
            #     if np.any(terminals):
            #         break

            timesteps_executed += self.num_environments
            env_frames += self.num_environments
            env_states = next_states
            current_iteration_time = time.perf_counter() - current_iteration_start_timestamp

            # Do accounting for each environment.
            state_buffer = np.array(self.preprocessed_states_buffer)
            for i, env_id in enumerate(self.env_ids):
                # Set is preprocessed to False because env_states are currently NOT preprocessed.
                self.is_preprocessed[env_id] = False
                current_episode_timesteps[i] += 1
                # Each position is the running episode reward of that episode. Add step reward.
                current_episode_rewards[i] += step_rewards[i]
                sample_states[env_id].append(state_buffer[i])
                sample_actions[env_id].append(actions[i])
                sample_rewards[env_id].append(step_rewards[i])
                sample_terminals[env_id].append(terminals[i])
                current_episode_sample_times[i] += current_iteration_time

                # Terminate and reset episode for that environment.
                if terminals[i] or (0 < max_timesteps_per_episode <= current_episode_timesteps[i]):
                    self.finished_episode_rewards[i].append(current_episode_rewards[i])
                    self.finished_episode_timesteps[i].append(current_episode_timesteps[i])
                    self.finished_episode_total_times[i].append(time.perf_counter() - current_episode_start_timestamps[i])
                    self.finished_episode_sample_times[i].append(current_episode_sample_times[i])
                    episodes_executed[i] += 1
                    self.episodes_executed += 1

                    env_sample_states = sample_states[env_id]
                    # Get next states for this environment's trajectory.
                    env_sample_next_states = env_sample_states[1:]

                    next_state = self.agent.state_space.force_batch(next_states[i])
                    if self.preprocessors[env_id] is not None:
                        next_state = self.preprocessors[env_id].preprocess(next_state)

                    # Extend because next state has a batch dim.
                    env_sample_next_states.extend(next_state)

                    # Post-process this trajectory via n-step discounting.
                    # print("processing terminal episode of length:", len(env_sample_states))
                    post_s, post_a, post_r, post_next_s, post_t = self._truncate_n_step(env_sample_states,
                        sample_actions[env_id], sample_rewards[env_id], env_sample_next_states,
                        sample_terminals[env_id], was_terminal=True)

                    # Append to final result trajectories.
                    batch_states.extend(post_s)
                    batch_actions.extend(post_a)
                    batch_rewards.extend(post_r)
                    batch_next_states.extend(post_next_s)
                    batch_terminals.extend(post_t)

                    # Reset running trajectory for this env.
                    sample_states[env_id] = []
                    sample_actions[env_id] = []
                    sample_rewards[env_id] = []
                    sample_terminals[env_id] = []

                    # Reset this environment and its pre-processor stack.
                    env_states[i] = self.vector_env.reset(i)
                    if self.preprocessors[env_id] is not None:
                        self.preprocessors[env_id].reset()
                        # This re-fills the sequence with the reset state.
                        state = self.agent.state_space.force_batch(env_states[i])
                        # Pre - process, add to buffer
                        self.preprocessed_states_buffer[i] = np.array(self.preprocessors[env_id].preprocess(state))
                        self.is_preprocessed[env_id] = True
                    current_episode_rewards[i] = 0
                    current_episode_timesteps[i] = 0
                    current_episode_start_timestamps[i] = time.perf_counter()
                    current_episode_sample_times[i] = 0.0

            if 0 < num_timesteps <= timesteps_executed or (break_on_terminal and np.any(terminals)):
                self.total_worker_steps += timesteps_executed
                break

        self.last_terminals = terminals
        self.last_states = env_states
        self.last_ep_rewards = current_episode_rewards
        self.last_ep_timesteps = current_episode_timesteps
        self.last_ep_start_timestamps = current_episode_start_timestamps
        self.last_ep_sample_times = current_episode_sample_times

        # We already accounted for all terminated episodes. This means we only
        # have to do accounting for any unfinished fragments.
        for i, env_id in enumerate(self.env_ids):
            # This env was not terminal -> need to process remaining trajectory
            if not terminals[i]:
                env_sample_states = sample_states[env_id]
                # Get next states for this environment's trajectory.
                env_sample_next_states = env_sample_states[1:]
                next_state = self.agent.state_space.force_batch(next_states[i])
                if self.preprocessors[env_id] is not None:
                    next_state = self.preprocessors[env_id].preprocess(next_state)
                    # This is the env state in the next call so avoid double preprocessing
                    # by adding to buffer.
                    self.preprocessed_states_buffer[i] = np.array(next_state)
                    self.is_preprocessed[env_id] = True

                # Extend because next state has a batch dim.
                env_sample_next_states.extend(next_state)
                post_s, post_a, post_r, post_next_s, post_t = self._truncate_n_step(env_sample_states,
                    sample_actions[env_id], sample_rewards[env_id], env_sample_next_states,
                    sample_terminals[env_id], was_terminal=False)

                batch_states.extend(post_s)
                batch_actions.extend(post_a)
                batch_rewards.extend(post_r)
                batch_next_states.extend(post_next_s)
                batch_terminals.extend(post_t)

        # Perform final batch-processing once.
        sample_batch, batch_size = self._batch_process_sample(batch_states, batch_actions,
                                                              batch_rewards, batch_next_states, batch_terminals)

        total_time = (time.monotonic() - start) or 1e-10
        self.sample_steps.append(timesteps_executed)
        self.sample_times.append(total_time)
        self.sample_env_frames.append(env_frames)

        # Note that the controller already evaluates throughput so there is no need
        # for each worker to calculate expensive statistics now.
        return EnvironmentSample(
            sample_batch=sample_batch,
            batch_size=batch_size,
            metrics=dict(
                runtime=total_time,
                # Agent act/observe throughput.
                timesteps_executed=timesteps_executed,
                ops_per_second=(timesteps_executed / total_time),
            )
        )

    @ray.method(num_return_vals=2)
    def execute_and_get_with_count(self):
        sample = self.execute_and_get_timesteps(num_timesteps=self.worker_sample_size)
        return sample, sample.batch_size

    def set_policy_weights(self, weights):
        self.agent.set_policy_weights(weights)

    def get_workload_statistics(self):
        """
        Returns performance results for this worker.

        Returns:
            dict: Performance metrics.
        """
        # Adjust env frames for internal env frameskip:
        adjusted_frames = [env_frames * self.env_frame_skip for env_frames in self.sample_env_frames]
        if len(self.finished_episode_rewards) > 0:
            all_finished_rewards = list()
            for env_reward_list in self.finished_episode_rewards:
                all_finished_rewards.extend(env_reward_list)
            min_episode_reward = np.min(all_finished_rewards)
            max_episode_reward = np.max(all_finished_rewards)
            mean_episode_reward = np.mean(all_finished_rewards)
            # Mean of final episode rewards over all envs
            final_episode_reward = np.mean([env_rewards[-1] for env_rewards in self.finished_episode_rewards])
        else:
            # Will be aggregated in executor.
            min_episode_reward = None
            max_episode_reward = None
            mean_episode_reward = None
            final_episode_reward = None

        return dict(
            episode_timesteps=self.finished_episode_timesteps,
            episode_rewards=self.finished_episode_rewards,
            episode_total_times=self.finished_episode_total_times,
            episode_sample_times=self.finished_episode_sample_times,
            min_episode_reward=min_episode_reward,
            max_episode_reward=max_episode_reward,
            mean_episode_reward=mean_episode_reward,
            final_episode_reward=final_episode_reward,
            episodes_executed=self.episodes_executed,
            worker_steps=self.total_worker_steps,
            mean_worker_ops_per_second=sum(self.sample_steps) / sum(self.sample_times),
            mean_worker_env_frames_per_second=sum(adjusted_frames) / sum(self.sample_times)
        )

    def _truncate_n_step(self, states, actions, rewards, next_states, terminals, was_terminal=True):
        """
        Computes n-step truncation for exactly one episode segment of one environment.

        Returns:
             n-step truncated (shortened) version.
        """
        if self.n_step_adjustment > 1:
            new_len = len(states) - self.n_step_adjustment + 1

            # There are 2 cases. If the trajectory did not end in a terminal,
            # we just have to move states forward and truncate.
            if was_terminal:
                # We know the ONLY last terminal is True.
                terminal_position = len(rewards) - 1
                for i in range(len(rewards)):
                    for j in range(1, self.n_step_adjustment):
                        # Outside sample data. Stop inner loop and set truncate = True
                        if i + j >= len(next_states):
                            break
                        # Normal case: No terminal ahead (so far) in n-step sequence.
                        if i + j < terminal_position:
                            next_states[i] = next_states[i + j]
                            rewards[i] += self.discount ** j * rewards[i + j]
                        # Terminal ahead: Don't go beyond it.
                        # Repeat it for the remaining n-steps and always assume r=0.0.
                        else:
                            next_states[i] = next_states[terminal_position]
                            terminals[i] = True
                            if i + j <= terminal_position:
                                rewards[i] += self.discount ** j * rewards[i + j]
            else:
                # We know this segment does not contain any terminals so we simply have to adjust next
                # states and rewards.
                for i in range_(len(rewards) - self.n_step_adjustment + 1):
                    for j in range_(1, self.n_step_adjustment):
                        next_states[i] = next_states[i + j]
                        rewards[i] += self.discount ** j * rewards[i + j]
                for arr in [states, actions, rewards, next_states, terminals]:
                    del arr[new_len:]

        return states, actions, rewards, next_states, terminals

    def _batch_process_sample(self, states, actions, rewards, next_states, terminals):
        """
        Batch Post-processes sample, e.g. by computing priority weights, and compressing.

        Args:
            states (list): List of states.
            actions (list): List of actions.
            rewards (list): List of rewards.
            next_states: (list): List of next_states.
            terminals (list): List of terminals.

        Returns:
            dict: Sample batch dict.
        """
        weights = np.ones_like(rewards)

        # Compute loss-per-item.
        if self.worker_computes_weights:
            # Next states were just collected, we batch process them here.
            # TODO make generic agent method?
            _, loss_per_item = self.agent.get_td_loss(
                dict(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    terminals=terminals,
                    next_states=next_states,
                    importance_weights=weights
                )
            )
            weights = np.abs(loss_per_item) + SMALL_NUMBER

        return dict(
            states=np.array([ray_compress(state) for state in states]),
            actions=np.array(actions),
            rewards=np.array(rewards),
            terminals=np.array(terminals),
            next_states=np.array([ray_compress(next_state) for next_state in next_states]),
            importance_weights=np.array(weights)
        ), len(rewards)

    def get_action(self, states, use_exploration, apply_preprocessing):
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
                                                    apply_preprocessing=apply_preprocessing)]
                else:
                    action = self.agent.get_action(states=states, use_exploration=use_exploration,
                                                    apply_preprocessing=apply_preprocessing)
            return action
        else:
            return self.agent.get_action(states=states, use_exploration=use_exploration,
                                         apply_preprocessing=apply_preprocessing)

