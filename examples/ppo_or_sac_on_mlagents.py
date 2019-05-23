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
Example script for training a Proximal policy optimization agent on an ML-Agents Env running locally in Unity3D.

Usage:

python ppo_or_sac_on_mlagents.py [--config ./configs/ppo_mlagents_[3dball_hard|banana_collector].json]

"""

import json
import os
import sys

import numpy as np
from absl import flags
from rlgraph.agents import Agent
from rlgraph.environments import MLAgentsEnv
from rlgraph.execution import SingleThreadedWorker

FLAGS = flags.FLAGS

# Use different configs here for either PPO or SAC algos.
# - ./configs/ppo_mlagents_banana_collector.json learns the BananaCollector Env and reaches the benchmark (10)
#   after about 1500 episodes using PPO with container actions.
# - ./configs/ppo_mlagents_3dball_hard.json learns the 3DBall (hard version) Env using PPO.
# - ./configs/sac_mlagents_3dball_hard.json learns the 3DBall (hard version) Env using SAC.
flags.DEFINE_string('config', './configs/ppo_mlagents_banana_collector.json', 'Agent config file.')


def main(argv):
    try:
        FLAGS(argv)
    except flags.Error as e:
        print('%s\\nUsage: %s ARGS\\n%s' % (e, sys.argv[0], FLAGS))

    agent_config_path = os.path.join(os.getcwd(), FLAGS.config)
    with open(agent_config_path, 'rt') as fp:
        agent_config = json.load(fp)

    env = MLAgentsEnv()

    agent = Agent.from_spec(
        agent_config,
        state_space=env.state_space,
        action_space=env.action_space
    )
    episode_returns = []

    def episode_finished_callback(episode_return, duration, timesteps, **kwargs):
        episode_returns.append(episode_return)
        finished_episodes = len(episode_returns)
        if finished_episodes % 4 == 0:
            print(
                "Episode {} finished in {:d}sec: total avg. reward={:.2f}; last 10 episodes={:.2f}; last "
                "100 episodes={:.2f}".format(
                    finished_episodes, int(duration), np.mean(episode_returns),
                    np.mean(episode_returns[-min(finished_episodes, 10):]),
                    np.mean(episode_returns[-min(finished_episodes, 100):])
                )
            )

    worker = SingleThreadedWorker(
        env_spec=env, agent=agent, render=False, worker_executes_preprocessing=False,
        episode_finish_callback=episode_finished_callback
    )
    print("Starting workload, this will take some time for the agents to build.")

    # Use exploration is true for training, false for evaluation.
    worker.execute_timesteps(500000, use_exploration=True)

    print("Mean reward: {:.2f} / over the last 10 episodes: {:.2f}".format(
        np.mean(episode_returns), np.mean(episode_returns[-10:])
    ))


if __name__ == '__main__':
    main(sys.argv)
