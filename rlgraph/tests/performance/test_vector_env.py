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
from copy import deepcopy

from six.moves import xrange as range_

from rlgraph.environments import Environment, SequentialVectorEnv


class TestVectorEnv(unittest.TestCase):
    """
    Tests environment throughput of Vector environment versus simple environment.
    """
    env_spec = dict(
        type="openai",
        gym_env="Pong-v0"
    )

    samples = 10000
    num_vector_envs = 4

    def test_individual_env(self):
        env = Environment.from_spec(self.env_spec)
        action_space = env.action_space

        env.reset()
        start = time.monotonic()
        ep_length = 0
        for _ in range_(self.samples):
            state, reward, terminal, info = env.step(action_space.sample())
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
        env = Environment.from_spec(deepcopy(self.env_spec))
        action_space = env.action_space

        vector_env.reset_all()
        start = time.monotonic()
        ep_lengths = [0 for _ in range_(self.num_vector_envs)]

        for _ in range_(self.samples):
            # Sample all envs at once.
            states, rewards, terminals, infos = vector_env.step(action_space.sample(
                size=self.num_vector_envs)
            )
            ep_lengths = [ep_length + 1 for ep_length in ep_lengths]
            for i, terminal in enumerate(terminals):
                if terminal:
                    print("reset env {} after {} states".format(i, ep_lengths[i]))
                    vector_env.reset(i)
                    ep_lengths[i] = 0

        runtime = time.monotonic() - start
        tp = self.samples / runtime

        print('Testing individual env {} performance:'.format(self.env_spec["gym_env"]))
        print('Ran {} steps, throughput: {} states/s, total time: {} s'.format(
            self.samples, tp, runtime
        ))
