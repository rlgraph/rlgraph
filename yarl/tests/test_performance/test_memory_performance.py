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

from yarl.components import ReplayMemory, PrioritizedReplay
from yarl.envs import OpenAIGymEnv
from yarl.spaces import Dict, BoolBox
from yarl.tests import ComponentTest


class TestMemoryPerformance(unittest.TestCase):

    # Note: Using Atari states here has to be done with care because without preprocessing, these will require
    # large amount sof memory.
    env = OpenAIGymEnv(gym_env='CartPole-v0')

    # Inserts.
    capacity = 100000
    inserts = 1000
    chunk_size = 64

    # Samples.
    samples = 1000
    sample_batch_size = 64

    alpha = 1.0
    beta = 1.0
    max_priority = 1.0

    def test_replay_insert(self):
        """
        Tests individual and chunked insert performance into replay memory.
        """

        record_space = Dict(
            states=self.env.state_space,
            actions=self.env.action_space,
            reward=float,
            terminals=BoolBox(),
            add_batch_rank=True
        )
        input_spaces = dict(
            insert_records=record_space,
            get_records=int
        )

        memory = ReplayMemory(
            capacity=self.capacity,
            next_states=True
        )
        test = ComponentTest(component=memory, input_spaces=input_spaces)

        records = [record_space.sample(size=1) for _ in range(self.inserts)]
        start = time.monotonic()
        for record in records:
            test.test(api_methods=dict(insert_records=record), expected_outputs=None)
        end = time.monotonic() - start

        tp = len(records) / end
        print('#### Testing Replay memory ####')
        print('Testing insert performance:')
        print('Inserted {} separate records, throughput: {} records/s, total time: {} s'.format(
            len(records), tp, end
        ))

        record_chunks = [record_space.sample(size=self.chunk_size) for _ in range(self.inserts)]
        start = time.monotonic()
        for chunk in record_chunks:
            test.test(api_methods=dict(insert_records=chunk), expected_outputs=None)
        end = time.monotonic() - start

        tp = len(record_chunks) * self.chunk_size / end
        print('Inserted {} record chunks of size {}, throughput: {} records/s, total time: {} s'.format(
            len(record_chunks), self.chunk_size, tp, end
        ))

        print('Testing sample performance:')
        start = time.monotonic()
        for _ in range(self.samples):
            test.test(api_methods=dict(get_records=self.sample_batch_size), expected_outputs=None)
        end = time.monotonic() - start
        tp = self.samples / end

        print('Sampled {} batches of size {}, throughput: {} sample-ops/s, total time: {} s'.format(
            self.samples, self.sample_batch_size, tp, end
        ))



    def test_prioritized_replay_insert(self):
        """
        Tests individual and chunked insert performance into replay memory.
        """

        record_space = Dict(
            states=self.env.state_space,
            actions=self.env.action_space,
            reward=float,
            terminals=BoolBox(),
            add_batch_rank=True
        )
        input_spaces = dict(
            insert_records=record_space,
            get_records=int
        )

        memory = PrioritizedReplay(
            capacity=self.capacity,
            next_states=True,
            alpha=self.alpha,
            beta=self.beta
        )
        test = ComponentTest(component=memory, input_spaces=input_spaces)

        records = [record_space.sample(size=1) for _ in range(self.inserts)]
        start = time.monotonic()
        for record in records:
            test.test(api_methods=dict(insert_records=record), expected_outputs=None)
        end = time.monotonic() - start

        tp = len(records) / end
        print('#### Testing Prioritized Replay memory ####')
        print('Testing insert performance:')
        print('Inserted {} separate records, throughput: {} records/s, total time: {} s'.format(
            len(records), tp, end
        ))

        record_chunks = [record_space.sample(size=self.chunk_size) for _ in range(self.inserts)]
        start = time.monotonic()
        for chunk in record_chunks:
            test.test(api_methods=dict(insert_records=chunk), expected_outputs=None)
        end = time.monotonic() - start

        tp = len(record_chunks) * self.chunk_size / end
        print('Inserted {} record chunks of size {}, throughput: {} records/s, total time: {} s'.format(
            len(record_chunks), self.chunk_size, tp, end
        ))

        print('Testing sample performance:')
        start = time.monotonic()
        for _ in range(self.samples):
            test.test(api_methods=dict(get_records=self.sample_batch_size), expected_outputs=None)
        end = time.monotonic() - start
        tp = self.samples / end

        print('Sampled {} batches of size {}, throughput: {} sample-ops/s, total time: {} s'.format(
            self.samples, self.sample_batch_size, tp, end
        ))
