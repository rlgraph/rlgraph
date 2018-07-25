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

from six.moves import xrange as range_
import numpy as np
import time

from rlgraph import SMALL_NUMBER
from rlgraph.backend_system import get_distributed_backend
from rlgraph.execution.environment_sample import EnvironmentSample
from rlgraph.execution.ray import RayExecutor
from rlgraph.execution.ray.ray_actor import RayActor
from rlgraph.execution.ray.ray_util import ray_compress

if get_distributed_backend() == "ray":
    import ray


@ray.remote
class RayWorker(RayActor):
    """
    Ray wrapper for single threaded worker, provides further api methods to interact
    with the agent used in the worker.
    """
    def __init__(self, agent_config, env_spec, worker_spec, frameskip=1):
        """
        Creates agent and environment for Ray worker.

        Args:
            agent_config (dict): Agent configuration dict.
            env_spec (dict): Environment config for environment to run.
            worker_spec (dict): Worker parameters.
            frameskip (int): How often actions are repeated after retrieving them from the agent.
        """
        # Should be set.
        assert get_distributed_backend() == "ray"

        # Ray cannot handle **kwargs in remote objects.
        self.environment = RayExecutor.build_env_from_config(env_spec)

        # Then update agent config.
        agent_config['state_space'] = self.environment.state_space
        agent_config['action_space'] = self.environment.action_space

        # Worker computes weights for prioritized sampling.
        self.worker_computes_weights = worker_spec.pop("worker_computes_weights", True)
        self.n_step_adjustment = worker_spec.pop("n_step_adjustment", 1)
        self.discount = agent_config.get("discount", 0.99)

        # Ray cannot handle **kwargs in remote objects.
        self.agent = self.setup_agent(agent_config, worker_spec)
        self.frameskip = frameskip

        # Save these so they can be fetched after training if desired.
        self.episode_rewards = []
        self.episode_timesteps = []
        self.total_worker_steps = 0
        self.episodes_executed = 0

        # Step time and steps done per call to execute_and_get to measure throughput of this worker.
        self.sample_times = []
        self.sample_steps = []
        self.sample_env_frames = []

        # To continue running through multiple exec calls.
        self.last_state = self.environment.reset()
        self.agent.reset()

        # Was the last state a terminal state so env should be reset in next call?
        self.last_terminal = False
        self.last_ep_timestep = 0
        self.last_ep_reward = 0

    def setup_agent(self, agent_config, worker_spec):
        """
        Sets up agent, potentially modifying its configuration via worker specific settings.
        """
        sample_exploration = worker_spec.pop("sample_exploration", False)
        # Adjust exploration for this worker.
        if sample_exploration:
            exploration_min_value = worker_spec.pop("exploration_min_value", 0.0)
            epsilon_spec = agent_config["exploration_spec"]["epsilon_spec"]

            if "decay_spec" in epsilon_spec:
                decay_from = epsilon_spec["decay_spec"]["from"]
                assert decay_from >= exploration_min_value, \
                    "Min value for exploration sampling must be smaller than" \
                    "decay_from {} in exploration_spec but is {}.".format(decay_from, exploration_min_value)

                # Sample a new initial epsilon from the interval [exploration_min_value, decay_from).
                sampled_from = np.random.uniform(low=exploration_min_value, high=decay_from)
                epsilon_spec["decay_spec"]["from"] = sampled_from

        return RayExecutor.build_agent_from_config(agent_config)

    # Remote functions to interact with this workers agent.
    def call_agent_op(self, op, inputs=None):
        self.agent.call_graph_op(op, inputs)

    def execute_and_get_timesteps(
        self,
        num_timesteps,
        max_timesteps_per_episode=0,
        use_exploration=True,
        break_on_terminal=False
    ):
        """
        Collects and returns timestep experience.

        Args:
            break_on_terminal (Optional[bool]): If true, breaks when a terminal is encountered. If false,
                executes exactly 'num_timesteps' steps.
        """
        start = time.monotonic()
        timesteps_executed = 0
        # Executed episodes within this exec call.
        episodes_executed = 0
        env_frames = 0
        states = []
        actions = []
        rewards = []
        next_states = []
        terminals = []

        while timesteps_executed < num_timesteps:
            # Reset Env and Agent either if finished an episode in current loop or if last state
            # from previous execution was terminal.
            if self.last_terminal is True or episodes_executed > 0:
                state = self.environment.reset()
                self.agent.reset()
                self.last_ep_reward = 0
                # The reward accumulated over one episode.
                episode_reward = 0
                episode_timestep = 0
            else:
                # Continue training between calls.
                state = self.last_state
                episode_reward = self.last_ep_reward
                episode_timestep = self.last_ep_timestep

            # Whether the episode has terminated.
            terminal = False
            while True:
                action, preprocessed_state = self.agent.get_action(states=state, use_exploration=use_exploration,
                                                                   extra_returns="preprocessed_states")
                states.append(state)
                actions.append(action)

                # Accumulate the reward over n env-steps (equals one action pick). n=self.frameskip.
                reward = 0
                next_state = None
                for _ in range_(self.frameskip):
                    next_state, step_reward, terminal, info = self.environment.step(actions=action)
                    env_frames += 1
                    reward += step_reward

                rewards.append(reward)
                terminals.append(terminal)
                next_states.append(next_state)
                episode_reward += reward
                timesteps_executed += 1
                episode_timestep += 1
                state = next_state

                if terminal or (0 < num_timesteps <= timesteps_executed) or \
                        (0 < max_timesteps_per_episode <= episode_timestep):
                    episodes_executed += 1
                    # Just return all samples collected so far.
                    self.episode_rewards.append(episode_reward)
                    self.episode_timesteps.append(episode_timestep)
                    self.total_worker_steps += timesteps_executed

                    if break_on_terminal:
                        self.last_terminal = True
                        total_time = (time.monotonic() - start) or 1e-10
                        self.sample_steps.append(timesteps_executed)
                        self.sample_times.append(total_time)
                        self.sample_env_frames.append(env_frames)
                        sample_batch = self._process_sample_if_necessary(states, actions,
                                                                         rewards, next_states, terminals)

                        return EnvironmentSample(
                            sample_batch,
                            metrics=dict(
                                # Just pass this to know later how this sample was configured.
                                break_on_terminal=break_on_terminal,
                                runtime=total_time,
                                # Agent act/observe throughput.
                                timesteps_executed=timesteps_executed,
                                ops_per_second=(timesteps_executed / total_time),
                                # Env frames including action repeats.
                                env_frames=env_frames,
                                env_frames_per_second=(env_frames / total_time),
                                episodes_executed=1,
                                episodes_per_minute=(1 / (total_time / 60)),
                                episode_rewards=episode_reward
                            )
                        )
                    else:
                        break

            self.episodes_executed += 1

        # Otherwise return when all time steps done
        self.last_terminal = terminal
        self.last_ep_reward = episode_reward

        total_time = (time.monotonic() - start) or 1e-10
        self.sample_steps.append(timesteps_executed)
        self.sample_times.append(total_time)
        self.sample_env_frames.append(env_frames)

        sample_batch = self._process_sample_if_necessary(states, actions, rewards, next_states, terminals)

        return EnvironmentSample(
            sample_batch,
            metrics=dict(
                break_on_terminal=break_on_terminal,
                runtime=total_time,
                # Agent act/observe throughput.
                timesteps_executed=timesteps_executed,
                ops_per_second=(timesteps_executed / total_time),
                # Env frames including action repeats.
                env_frames=env_frames,
                env_frames_per_second=(env_frames / total_time),
                episodes_executed=self.episodes_executed,
                episodes_per_minute=(1 / (total_time / 60)),
                episode_rewards=self.episode_rewards,
            )
        )

    def set_policy_weights(self, weights):
        self.agent.set_policy_weights(weights)

    def get_workload_statistics(self):
        """
        Returns performance results for this worker.

        Returns:
            dict: Performance metrics.
        """
        return dict(
            episode_timesteps=self.episode_timesteps,
            episode_rewards=self.episode_rewards,
            min_episode_reward=np.min(self.episode_rewards),
            max_episode_reward=np.max(self.episode_rewards),
            mean_episode_reward=np.mean(self.episode_rewards),
            final_episode_reward=self.episode_rewards[-1],
            episodes_executed=self.episodes_executed,
            worker_steps=self.total_worker_steps,
            mean_worker_ops_per_second=sum(self.sample_steps) / sum(self.sample_times),
            mean_worker_env_frames_per_second=sum(self.sample_env_frames) / sum(self.sample_times)
        )

    def _process_sample_if_necessary(self, states, actions, rewards, next_states, terminals):
        """
        Post-processes sample, e.g. by computing priority weights, compressing, applying
        n-step corrections, ported from ray RLLib.

        Args:
            states (list): List of states.
            actions (list): List of actions.
            rewards (list): List of rewards.
            next_states: (list): List of next_states.
            terminals (list): List of terminals.

        Returns:
            dict: Sample batch dict.
        """
        weights = np.ones_like(terminals)

        if self.n_step_adjustment > 1:
            for i in range_(len(rewards) - self.n_step_adjustment + 1):
                # Ignore terminals.
                if terminals[i]:
                    continue
                for j in range_(1, self.n_step_adjustment):
                    states[i] = states[i + j]
                    rewards[i] += self.discount ** j * rewards[i + j]

                    # Set remaining reward to 0.
                    if terminals[i + j]:
                        break

            # Truncate.
            new_len = len(states) - self.n_step_adjustment + 1
            for arr in [states, actions, rewards, next_states, terminals]:
                del arr[new_len:]

        # Compute loss-per-item.
        if self.worker_computes_weights:
            _, loss_per_item = self.agent.update(
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
            states=[ray_compress(state) for state in states],
            actions=actions,
            rewards=rewards,
            terminals=terminals,
            importance_weights=weights
        )
