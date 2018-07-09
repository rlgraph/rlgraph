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

import unittest

import time

from ray.rllib.optimizers.replay_buffer import PrioritizedReplayBuffer

from yarl.components import ReplayMemory, PrioritizedReplay
from yarl.components.memories.mem_prioritized_replay import MemPrioritizedReplay
from yarl.envs import OpenAIGymEnv
from yarl.spaces import Dict, BoolBox, FloatBox, IntBox
from yarl.tests import ComponentTest


class TestPythonMemoryPerformance(unittest.TestCase):

    # Note: Using Atari states here has to be done with care because without preprocessing, these will require
    # large amount sof memory.
    env = OpenAIGymEnv(gym_env='CartPole-v0')

    # Inserts.
    capacity = 100000
    inserts = 1000000

    # Samples.
    samples = 1000
    sample_batch_size = 64

    alpha = 1.0
    beta = 1.0
    max_priority = 1.0

    def test_ray_prioritized_replay(self):
        """
        Tests Ray's memory performance.
        """
        memory = PrioritizedReplayBuffer(
            size=self.capacity,
            alpha=1.0,
            clip_rewards=True
        )
        # Testing insert performance
        record_space = Dict(
            states=self.env.state_space,
            actions=self.env.action_space,
            reward=float,
            terminals=BoolBox(),
            add_batch_rank=True
        )
        records = [record_space.sample(size=1) for _ in range(self.inserts)]

        start = time.monotonic()
        for record in records:
            memory.add(
                obs_t=record['states'],
                action=record['actions'],
                reward=record['reward'],
                obs_tp1=record['states'],
                done=record['terminals'],
                weight=None
            )
        end = time.monotonic() - start
        tp = len(records) / end
        print('#### Testing Ray Prioritized Replay memory ####')
        print('Testing insert performance:')
        print('Inserted {} separate records, throughput: {} records/s, total time: {} s'.format(
            len(records), tp, end
        ))

    def test_yarl_prioritized_replay(self):
        """
        Tests Yarl's python memory performance.
        """
        memory = MemPrioritizedReplay(
            capacity=self.capacity,
            alpha=1.0
        )
        # Testing insert performance
        record_space = Dict(
            states=self.env.state_space,
            actions=self.env.action_space,
            reward=float,
            terminals=BoolBox(),
            add_batch_rank=True
        )
        records = [record_space.sample(size=1) for _ in range(self.inserts)]

        start = time.monotonic()
        for record in records:
            memory.insert_records(record)
        end = time.monotonic() - start
        tp = len(records) / end

        print('#### Testing YARL python prioritized replay ####')
        print('Testing insert performance:')
        print('Inserted {} separate records, throughput: {} records/s, total time: {} s'.format(
            len(records), tp, end
        ))

