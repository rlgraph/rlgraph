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
Example script for training a single-node IMPALA [1] agent on an OpenAI gym environment.
A single-node agent uses multi-threading (via tf's queue runners) to collect experiences (using
the "mu"-policy) and a learner (main) thread to update the model (the "pi"-policy).

Usage:

python impala_openai_gym_with_lstm.py [--config configs/impala_openai_gym_with_lstm.json] [--env LunarLander-v2]?

[1] IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures - Espeholt, Soyer,
    Munos et al. - 2018 (https://arxiv.org/abs/1802.01561)
"""

import json
import os
import sys

from absl import flags
import numpy as np
import time

from rlgraph.agents import Agent
from rlgraph.environments import OpenAIGymEnv


FLAGS = flags.FLAGS

flags.DEFINE_string('config', './configs/impala_openai_gym_with_lstm.json', 'Agent config file.')
flags.DEFINE_string('env', None, 'openAI gym environment ID.')
flags.DEFINE_integer('visualize', -1, 'Show training for n worker(s).')


def main(argv):
    try:
        FLAGS(argv)
    except flags.Error as e:
        print('%s\\nUsage: %s ARGS\\n%s' % (e, sys.argv[0], FLAGS))

    agent_config_path = os.path.join(os.getcwd(), FLAGS.config)
    with open(agent_config_path, 'rt') as fp:
        agent_config = json.load(fp)

    # Override openAI gym env per command line.
    if FLAGS.env is None:
        env_spec = agent_config["environment_spec"]
    else:
        env_spec = dict(type="openai-gym", gym_env=FLAGS.env)
    # Override number of visualized envs per command line.
    if FLAGS.visualize != -1:
        env_spec["visualize"] = FLAGS.visualize

    dummy_env = OpenAIGymEnv.from_spec(env_spec)
    agent = Agent.from_spec(
        agent_config,
        state_space=dummy_env.state_space,
        action_space=dummy_env.action_space
    )
    dummy_env.terminate()

    learn_updates = 6000
    mean_returns = []
    for i in range(learn_updates):
        ret = agent.update()
        mean_return = _calc_mean_return(ret)
        mean_returns.append(mean_return)
        print("Iteration={} Loss={:.4f} Avg-reward={:.2f}".format(i, float(ret[1]), mean_return))

    print("Mean return: {:.2f} / over the last 10 episodes: {:.2f}".format(
        np.nanmean(mean_returns), np.nanmean(mean_returns[-10:])
    ))

    time.sleep(1)
    agent.terminate()
    time.sleep(3)


def _calc_mean_return(records):
    size = records[3]["rewards"].size
    rewards = records[3]["rewards"].reshape((size,))
    terminals = records[3]["terminals"].reshape((size,))
    returns = list()
    return_ = 0.0
    for r, t in zip(rewards, terminals):
        return_ += r
        if t:
            returns.append(return_)
            return_ = 0.0

    return np.nanmean(returns)


if __name__ == '__main__':
    main(sys.argv)
