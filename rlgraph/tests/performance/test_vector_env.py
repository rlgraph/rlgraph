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
import unittest
import numpy as np

from rlgraph.agents import Agent
from rlgraph.tests.test_util import config_from_path
from six.moves import xrange as range_

from rlgraph.environments import Environment, SequentialVectorEnv


class TestVectorEnv(unittest.TestCase):
    """
    Tests environment throughput of Vector environment versus simple environment.
    """
    env_spec = dict(
        type="openai",
        gym_env="Pong-v0",
        frameskip=4,
        max_num_noops=30,
        episodic_life=True
    )

    samples = 50000
    num_vector_envs = 4

    def test_individual_env(self):
        env = Environment.from_spec(self.env_spec)
        agent = Agent.from_spec(
            # Uses 2015 DQN parameters as closely as possible.
            config_from_path("configs/dqn_agent_for_pong.json"),
            state_space=env.state_space,
            # Try with "reduced" action space (actually only 3 actions, up, down, no-op)
            action_space=env.action_space
        )

        state = env.reset()
        start = time.monotonic()
        ep_length = 0
        for _ in range_(self.samples):
            action = agent.get_action(state)
            state, reward, terminal, info = env.step(action)

            ep_length += 1
            if terminal:
                print("reset after {} states".format(ep_length))
                env.reset()
                ep_length = 0

        runtime = time.monotonic() - start
        tp = self.samples / runtime

        print('Testing individual env {} performance:'.format(self.env_spec["gym_env"]))
        print('Ran {} steps, throughput: {} states/s, total time: {} s'.format(
            self.samples, tp, runtime
        ))

    def test_sequential_vector_env(self):
        vector_env = SequentialVectorEnv(
            num_envs=self.num_vector_envs,
            env_spec=self.env_spec,
            num_background_envs=2
        )
        agent = Agent.from_spec(
            # Uses 2015 DQN parameters as closely as possible.
            config_from_path("configs/dqn_vector_env.json"),
            state_space=vector_env.state_space,
            # Try with "reduced" action space (actually only 3 actions, up, down, no-op)
            action_space=vector_env.action_space
        )

        states = vector_env.reset_all()
        start = time.monotonic()
        ep_lengths = [0 for _ in range_(self.num_vector_envs)]

        for _ in range_(int(self.samples/ self.num_vector_envs)):
            # Sample all envs at once.
            actions, preprocessed_states = agent.get_action(states, extra_returns="preprocessed_states")
            states, rewards, terminals, infos = vector_env.step(actions)
            ep_lengths = [ep_length + 1 for ep_length in ep_lengths]

            for i, terminal in enumerate(terminals):
                if terminal:
                    print("reset env {} after {} states".format(i, ep_lengths[i]))
                    vector_env.reset(i)
                    ep_lengths[i] = 0

        runtime = time.monotonic() - start
        tp = self.samples / runtime

        print('Testing vector env {} performance:'.format(self.env_spec["gym_env"]))
        print('Ran {} steps, throughput: {} states/s, total time: {} s'.format(
            self.samples, tp, runtime
        ))
