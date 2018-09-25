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
from rlgraph.spaces import Dict, FloatBox, BoolBox, IntBox
from rlgraph.tests.test_util import config_from_path
from rlgraph.environments import Environment


class TestMultiGPUUpdates(unittest.TestCase):
    """
    Tests multi gpu update throughput.
    """
    env_spec = dict(
        type="openai",
        gym_env="Pong-v0",
        frameskip=4,
        max_num_noops=30,
        episodic_life=True
    )

    def test_update_throughput(self):
        env = Environment.from_spec(self.env_spec)
        # TODO comment in for multi gpu
        # config_from_path("configs/multi_gpu_ray_apex_for_pong.json"),
        config = config_from_path("configs/ray_apex_for_pong.json")

        # Adjust to usable GPUs for test system.
        num_gpus = [1]
        for gpu_count in num_gpus:
            config["execution_spec"]["gpu_spec"]["num_gpus"] = gpu_count
            config["execution_spec"]["gpu_spec"]["per_process_gpu_memory_fraction"] = 1.0 / gpu_count

            agent = Agent.from_spec(
                # TODO replace with config from above
                config_from_path("configs/ray_apex_for_pong.json"),
                state_space=env.state_space,
                # Try with "reduced" action space (actually only 3 actions, up, down, no-op)
                action_space=env.action_space
            )

            batch_space = Dict(
                states=agent.preprocessed_state_space,
                actions=env.action_space,
                rewards=FloatBox(),
                next_states=agent.preprocessed_state_space,
                terminals=IntBox(low=0, high=1),
                importance_weights=FloatBox(),
                add_batch_rank=True
            )

            batch_size = 512 * gpu_count
            num_samples = 50
            samples = [batch_space.sample(batch_size) for _ in range(num_samples)]

            times = []
            throughputs = []
            for sample in samples:
                start = time.perf_counter()
                agent.update(sample)
                runtime = time.perf_counter() - start
                times.append(runtime)
                throughputs.append(batch_size / runtime)

            print("Throughput: {} samples / s ({}) for {} GPUs".format(np.mean(throughputs),
                                                                       np.std(throughputs), gpu_count))
