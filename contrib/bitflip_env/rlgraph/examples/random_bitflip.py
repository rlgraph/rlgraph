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

"""
Example script for training a DQN agent on an OpenAI gym environment.

Usage:

python dqn_cartpole.py [--config configs/dqn_cartpole.json] [--env CartPole-v0]

```
# Run script
python dqn_cartpole.py
```
"""

import json
import os
import sys

import numpy as np
from absl import flags

from rlgraph.agents import Agent
from contrib.bitflip_env.rlgraph.environments import OpenAIGymEnv
from rlgraph.execution import SingleThreadedWorker

FLAGS = flags.FLAGS

flags.DEFINE_string('config', './configs/random_bitflip.json', 'Agent config file.')
flags.DEFINE_string('env', 'bitflip-v0', 'gym environment ID.')
flags.DEFINE_boolean('render', True, 'Render the environment.')
flags.DEFINE_integer('episodes', 200, 'Number of training episodes.')


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
        "gym_env": FLAGS.env
    })

    agent = Agent.from_spec(
        agent_config,
        state_space=env.state_space,
        action_space=env.action_space
    )

    rewards = []

    def episode_finished_callback(reward, duration, timesteps, **kwargs):
        rewards.append(reward)
        if len(rewards) % 10 == 0:
            print("Episode {} finished: reward={:.2f}, average reward={:.2f}.".format(
                len(rewards), reward, np.mean(rewards[-10:])
            ))

    worker = SingleThreadedWorker(env_spec=lambda: env, agent=agent, render=FLAGS.render,
                                  worker_executes_preprocessing=False,
                                  episode_finish_callback=episode_finished_callback)
    print("Starting workload, this will take some time for the agents to build.")
    results = worker.execute_episodes(FLAGS.episodes)

    print("Mean reward: {:.2f} / over the last 10 episodes: {:.2f}".format(
        np.mean(rewards), np.mean(rewards[-10:])
    ))


if __name__ == '__main__':
    main(sys.argv)
