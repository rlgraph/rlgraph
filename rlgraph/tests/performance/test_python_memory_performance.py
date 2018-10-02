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

import unittest

import numpy as np
import time

from six.moves import xrange as range_

from rlgraph import get_distributed_backend
from rlgraph.execution.ray.apex.apex_memory import ApexMemory
from rlgraph.execution.ray.ray_util import ray_compress
from rlgraph.spaces import Dict, BoolBox, FloatBox

if get_distributed_backend() == "ray":
    from ray.rllib.optimizers.replay_buffer import PrioritizedReplayBuffer


class TestPythonMemoryPerformance(unittest.TestCase):
    record_space = Dict(
        states=FloatBox(shape=(4,)),
        actions=FloatBox(shape=(2,)),
        reward=float,
        terminals=BoolBox(),
        add_batch_rank=True
    )

    # Apex params
    capacity = 2000000
    chunksize = 64
    inserts = 1000000

    # Samples.
    samples = 10000
    sample_batch_size = 50

    alpha = 0.6
    beta = 0.4
    max_priority = 1.0

    def test_ray_prioritized_replay_insert(self):
        """
        Tests Ray's memory performance.
        """
        assert get_distributed_backend() == "ray"
        memory = PrioritizedReplayBuffer(
            size=self.capacity,
            alpha=1.0,
            clip_rewards=True
        )
        # Test individual inserts.
        records = [self.record_space.sample(size=1) for _ in range_(self.inserts)]

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

        memory = PrioritizedReplayBuffer(
            size=self.capacity,
            alpha=1.0,
            clip_rewards=True
        )

        # Test chunked inserts -> done via external for loop in Ray.
        chunks = int(self.inserts / self.chunksize)
        records = [self.record_space.sample(size=self.chunksize) for _ in range_(chunks)]
        start = time.monotonic()
        for chunk in records:
            for i in range_(self.chunksize):
                memory.add(
                    obs_t=chunk['states'][i],
                    action=chunk['actions'][i],
                    reward=chunk['reward'][i],
                    obs_tp1=chunk['states'][i],
                    done=chunk['terminals'][i],
                    weight=None
                )
        end = time.monotonic() - start
        tp = len(records) * self.chunksize / end
        print('Testing chunked insert performance:')
        print('Inserted {} chunks, throughput: {} records/s, total time: {} s'.format(
            len(records), tp, end
        ))

    def test_ray_sampling(self):
        """
        Tests Ray's memory performance.
        """
        assert get_distributed_backend() == "ray"
        memory = PrioritizedReplayBuffer(
            size=self.capacity,
            alpha=1.0,
            clip_rewards=True
        )
        records = [self.record_space.sample(size=1) for _ in range_(self.inserts)]
        for record in records:
            memory.add(
                obs_t=ray_compress(record['states']),
                action=record['actions'],
                reward=record['reward'],
                obs_tp1=ray_compress(record['states']),
                done=record['terminals'],
                weight=None
            )
        start = time.monotonic()
        for _ in range_(self.samples):
            batch_tuple = memory.sample(self.sample_batch_size, beta=1.0)
        end = time.monotonic() - start
        tp = self.samples / end
        print('#### Testing Ray Prioritized Replay memory ####')
        print('Testing sampling performance:')
        print('Sampled {} batches, throughput: {} samples/s, total time: {} s'.format(
            self.samples, tp, end
        ))

    def test_ray_updating(self):
        """
        Tests Ray's memory performance.
        """
        assert get_distributed_backend() == "ray"
        memory = PrioritizedReplayBuffer(
            size=self.capacity,
            alpha=1.0,
            clip_rewards=True
        )
        records = [self.record_space.sample(size=1) for _ in range_(self.inserts)]
        for record in records:
            memory.add(
                obs_t=record['states'],
                action=record['actions'],
                reward=record['reward'],
                obs_tp1=record['states'],
                done=record['terminals'],
                weight=None
            )
        loss_values = [np.random.random(size=self.sample_batch_size) for _ in range_(self.samples)]
        indices = [np.random.randint(low=0, high=self.inserts, size=self.sample_batch_size) for _
                   in range_(self.samples)]

        start = time.monotonic()
        for index, loss in zip(indices, loss_values):
            memory.update_priorities(index, loss)
        end = time.monotonic() - start
        tp = len(indices) / end
        print('#### Testing Ray Prioritized Replay memory ####')
        print('Testing updating performance:')
        print('Updates {} loss batches, throughput: {} updates/s, total time: {} s'.format(
            len(indices), tp, end
        ))

    def test_rlgraph_apex_insert(self):
        """
        Tests RLgraph's python memory performance.
        """
        memory = ApexMemory(
            capacity=self.capacity,
            alpha=1.0
        )
        # Testing insert performance
        records = [self.record_space.sample(size=1) for _ in range(self.inserts)]

        start = time.monotonic()
        for record in records:
            memory.insert_records((
                 record['states'],
                 record['actions'],
                 record['reward'],
                 record['terminals'],
                 None
            ))
        end = time.monotonic() - start
        tp = len(records) / end

        print('#### Testing RLGraph python prioritized replay ####')
        print('Testing insert performance:')
        print('Inserted {} separate records, throughput: {} records/s, total time: {} s'.format(
            len(records), tp, end
        ))

        memory = ApexMemory(
            capacity=self.capacity,
            alpha=1.0
        )
        chunks = int(self.inserts / self.chunksize)
        records = [self.record_space.sample(size=self.chunksize) for _ in range_(chunks)]
        start = time.monotonic()
        for chunk in records:
            for i in range_(self.chunksize):
                memory.insert_records((
                    chunk['states'][i],
                    chunk['actions'][i],
                    chunk['reward'][i],
                    chunk['terminals'][i],
                    None
                ))

        end = time.monotonic() - start
        tp = len(records) * self.chunksize / end
        print('Testing chunked insert performance:')
        print('Inserted {} chunks, throughput: {} records/s, total time: {} s'.format(
            len(records), tp, end
        ))

    def test_rlgraph_sampling(self):
        """
        Tests RLgraph's sampling performance.
        """
        memory = ApexMemory(
            capacity=self.capacity,
            alpha=1.0
        )

        records = [self.record_space.sample(size=1) for _ in range_(self.inserts)]
        for record in records:
            memory.insert_records((
                 ray_compress(record['states']),
                 record['actions'],
                 record['reward'],
                 record['terminals'],
                 None
            ))
        start = time.monotonic()
        for _ in range_(self.samples):
            batch_tuple = memory.get_records(self.sample_batch_size)
        end = time.monotonic() - start
        tp = self.samples / end
        print('#### Testing RLGraph Prioritized Replay memory ####')
        print('Testing sampling performance:')
        print('Sampled {} batches, throughput: {} batches/s, total time: {} s'.format(
            self.samples, tp, end
        ))

    def test_rlgraph_updating(self):
        """
        Tests RLGraph's memory performance.
        """
        memory = ApexMemory(
            capacity=self.capacity,
            alpha=1.0
        )

        records = [self.record_space.sample(size=1) for _ in range_(self.inserts)]
        for record in records:
            memory.insert_records((
                 record['states'],
                 record['actions'],
                 record['reward'],
                 record['terminals'],
                 None
            ))
        loss_values = [np.random.random(size=self.sample_batch_size) for _ in range_(self.samples)]
        indices = [np.random.randint(low=0, high=self.inserts, size=self.sample_batch_size) for _
                   in range_(self.samples)]

        start = time.monotonic()
        for index, loss in zip(indices, loss_values):
            memory.update_records(index, loss)
        end = time.monotonic() - start
        tp = len(indices) / end
        print('#### Testing RLGraph Prioritized Replay memory ####')
        print('Testing updating performance:')
        print('Updates {} loss batches, throughput: {} updates/s, total time: {} s'.format(
            len(indices), tp, end
        ))

    def test_ray_combined_ops(self):
        """
        Tests a combined workflow of insert, sample, update on the prioritized replay memory.
        """
        assert get_distributed_backend() == "ray"
        memory = PrioritizedReplayBuffer(
            size=self.capacity,
            alpha=1.0,
            clip_rewards=True
        )
        chunksize = 32

        # Test chunked inserts -> done via external for loop in Ray.
        chunks = int(self.inserts / chunksize)
        records = [self.record_space.sample(size=chunksize) for _ in range_(chunks)]
        loss_values = [np.random.random(size=self.sample_batch_size) for _ in range_(chunks)]
        start = time.monotonic()

        for chunk, loss_values in zip(records, loss_values):
            # Insert.
            for i in range_(chunksize):
                memory.add(
                    obs_t=ray_compress(chunk['states'][i]),
                    action=chunk['actions'][i],
                    reward=chunk['reward'][i],
                    obs_tp1=ray_compress(chunk['states'][i]),
                    done=chunk['terminals'][i],
                    weight=None
                )
            # Sample.
            batch_tuple = memory.sample(self.sample_batch_size, beta=1.0)
            indices = batch_tuple[-1]
            # Update
            memory.update_priorities(indices, loss_values)

        end = time.monotonic() - start
        tp = len(records) / end
        print('Ray: testing combined insert/sample/update performance:')
        print('Ran {} combined ops, throughput: {} combined ops/s, total time: {} s'.format(
            len(records), tp, end
        ))

    def test_rlgraph_combined_ops(self):
        """
        Tests a combined workflow of insert, sample, update on the prioritized replay memory.
        """
        memory = ApexMemory(
            capacity=self.capacity,
            alpha=1.0
        )

        chunksize = 32
        chunks = int(self.inserts / chunksize)
        records = [self.record_space.sample(size=chunksize) for _ in range_(chunks)]
        loss_values = [np.random.random(size=self.sample_batch_size) for _ in range_(chunks)]

        start = time.monotonic()
        for chunk, loss_values in zip(records, loss_values):
            # Each record now is a chunk.
            for i in range_(chunksize):
                memory.insert_records((
                    ray_compress(chunk['states'][i]),
                    chunk['actions'][i],
                    chunk['reward'][i],
                    chunk['terminals'][i],
                    None
                ))
            batch, indices, weights = memory.get_records(self.sample_batch_size)
            memory.update_records(indices, loss_values)

        end = time.monotonic() - start
        tp = len(records) / end
        print('RLGraph: Testing combined op performance:')
        print('Ran {} combined ops, throughput: {} combined ops/s, total time: {} s'.format(
            len(records), tp, end
        ))
