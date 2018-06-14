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

from six.moves import xrange
import time

from yarl.backend_system import get_distributed_backend

from yarl.execution import Worker
from yarl.execution.env_sample import EnvSample
from yarl.execution.ray import RayExecutor

if get_distributed_backend() == "ray":
    import ray


@ray.remote
class RayWorker(object):
    """
    Ray wrapper for single threaded worker, provides further api methods to interact
    with the agent used in the worker.
    """
    def __init__(self, env_spec, agent_config, repeat_actions=1):
        """
        Creates agent and environment for Ray worker.
        Args:
            env_spec (dict): Environment config for environment to run.
            agent_config (dict): Agent configuration dict.
            repeat_actions (int): How often actions are repeated after retrieving them from the agent.
        """
        # Should be set.
        assert get_distributed_backend() == "ray"

        # Ray cannot handle **kwargs in remote objects.
        self.environment = RayExecutor.build_env_from_config(env_spec)

        # Then update agent config.
        agent_config['state_space'] = self.environment.state_space
        agent_config['action_space'] = self.environment.action_space

        # Ray cannot handle **kwargs in remote objects.
        self.agent = RayExecutor.build_agent_from_config(agent_config)
        self.repeat_actions = repeat_actions

    # Remote functions to interact with this workers agent.
    def call_agent_op(self, op, inputs=None):
        self.agent.call_graph_op(op, inputs)

    def execute_and_get_timesteps(
        self,
        num_timesteps,
        max_timesteps_per_episode=0,
        deterministic=False,
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
        episodes_executed = 0
        env_frames = 0
        states = []
        actions = []
        rewards = []
        terminals = []
        episode_rewards = []
        state = self.environment.reset()

        while timesteps_executed < num_timesteps:
            # The reward accumulated over one episode.
            episode_reward = 0

            # Whether the episode has terminated.
            terminal = False
            while True:
                action = self.agent.get_action(states=state, deterministic=deterministic)

                states.append(state)
                actions.append(action)

                # Accumulate the reward over n env-steps (equals one action pick). n=self.repeat_actions
                reward = 0
                for _ in xrange(self.repeat_actions):
                    state, step_reward, terminal, info = self.environment.step(actions=action)
                    env_frames += 1
                    reward += step_reward

                rewards.append(reward)
                terminals.append(terminal)

                if terminal:
                    # Just return all samples collected so far.
                    if break_on_terminal:
                        total_time = (time.monotonic() - start) or 1e-10
                        return EnvSample(
                            states=states,
                            actions=actions,
                            rewards=rewards,
                            terminals=terminals,
                            metrics=dict(
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

            episodes_executed += 1
            episode_rewards.append(episode_reward)

        # Otherwise return when all time steps done
        total_time = (time.monotonic() - start) or 1e-10

        return EnvSample(
            states=states,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
            metrics=dict(
                runtime=total_time,
                # Agent act/observe throughput.
                timesteps_executed=timesteps_executed,
                ops_per_second=(timesteps_executed / total_time),
                # Env frames including action repeats.
                env_frames=env_frames,
                env_frames_per_second=(env_frames / total_time),
                episodes_executed=episodes_executed,
                episodes_per_minute=(1 / (total_time / 60)),
                episode_rewards=episode_rewards,
            )
        )

    # TODO decide later if using separate methods here for apex
    # def set_weights(self, weights):
    #     self.agent.call_graph_op("set_weights", weights)
    #
    # def get_batch(self):
    #     return self.agent.call_graph_op("sample")
