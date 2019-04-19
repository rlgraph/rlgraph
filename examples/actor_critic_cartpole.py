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

"""
Example script for training an actor-critic policy gradient agent on an OpenAI gym environment.

Usage:

python actor_critic_cartpole.py [--config configs/actor_critic_cartpole.json] [--env CartPole-v0]

```
# Run script
python actor_critic_cartpole.py
```
"""

import json
import os
import sys

import numpy as np
from absl import flags

from rlgraph.agents import Agent
from rlgraph.environments import OpenAIGymEnv
from rlgraph.execution import SingleThreadedWorker

FLAGS = flags.FLAGS

flags.DEFINE_string('config', './configs/actor_critic_cartpole.json', 'Agent config file.')
flags.DEFINE_string('env', 'CartPole-v0', 'openAI Gym environment ID.')
flags.DEFINE_bool('visualize', False, 'Whether to display the env during learning.')


def main(argv):
    try:
        FLAGS(argv)
    except flags.Error as e:
        print('%s\\nUsage: %s ARGS\\n%s' % (e, sys.argv[0], FLAGS))

    agent_config_path = os.path.join(os.getcwd(), FLAGS.config)
    with open(agent_config_path, 'rt') as fp:
        agent_config = json.load(fp)

    env = OpenAIGymEnv.from_spec({
        "type": "openai",
        "gym_env": FLAGS.env,
        "visualize": FLAGS.visualize
    })

    agent = Agent.from_spec(
        agent_config,
        state_space=env.state_space,
        action_space=env.action_space
    )

    episode_returns = []

    def episode_finished_callback(episode_return, duration, timesteps, **kwargs):
        episode_returns.append(episode_return)
        if len(episode_returns) % 10 == 0:
            print("Episode {} finished: reward={:.2f}, average reward={:.2f}.".format(
                len(episode_returns), episode_return, np.mean(episode_returns[-10:])
            ))

    worker = SingleThreadedWorker(env_spec=lambda: env, agent=agent, render=False, worker_executes_preprocessing=False,
                                  episode_finish_callback=episode_finished_callback)
    print("Starting workload, this will take some time for the agents to build.")

    # Use exploration is true for training, false for evaluation.
    worker.execute_timesteps(20000, use_exploration=True)

    # Note: A basic actor critic is very sensitive to hyper-parameters and might collapse after reaching the maximum
    # reward. In practice, it would be recommended to stop training when a reward threshold is reached.
    print("Mean reward: {:.2f} / over the last 10 episodes: {:.2f}".format(
        np.mean(episode_returns), np.mean(episode_returns[-10:])
    ))


if __name__ == '__main__':
    main(sys.argv)
