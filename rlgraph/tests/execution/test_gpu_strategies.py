# Copyright 2018 The RLgraph authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, softwamre
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import multiprocessing
from rlgraph.utils import root_logger
from logging import DEBUG
import time

from rlgraph.agents import ApexAgent
from rlgraph.components.layers.nn.lstm_layer import LSTMLayer
from rlgraph.environments import OpenAIGymEnv, RandomEnv
#from rlgraph.execution.ray import ApexExecutor
from rlgraph.spaces import IntBox, FloatBox
from rlgraph.tests.component_test import ComponentTest
from rlgraph.tests.test_util import config_from_path


def dummy_func(env_spec, cluster_spec):
    env = RandomEnv.from_spec(env_spec)
    state_space = env.state_space.with_extra_ranks(add_time_rank=True, add_batch_rank=True)
    lstm = LSTMLayer(units=50, time_major=True, device=dict(ops="/job:b/task:0/cpu", variables="/job:a/task:0/cpu"))
    test = ComponentTest(
        lstm,
        input_spaces=dict(inputs=state_space),
        action_space=env.action_space,
        execution_spec=dict(
            mode="distributed",
            gpu_spec=dict(
                gpus_enabled=True,
                max_usable_gpus=1,
                num_gpus=1
            ),
            distributed_spec=dict(job="b", task_index=0, cluster_spec=cluster_spec),
            session_config=dict(
                type="monitored-training-session",
                allow_soft_placement=True,
                log_device_placement=True
            ),
            enable_timeline=True,
        )
    )
    print("generated env and lstm with shared variables: sleeping")
    time.sleep(120)


class TestGpuStrategies(unittest.TestCase):
    """
    Tests gpu strategies.
    """
    env_spec = dict(
        type="openai",
        gym_env="PongNoFrameskip-v4",
        # The frameskip in the agent config will trigger worker skips, this
        # is used for internal env.
        frameskip=4,
        max_num_noops=30,
        episodic_life=True
    )
    cluster_spec = dict(a=["localhost:22222"], b=["localhost:22223"])

    def test_multi_gpu_agent_compilation(self):
        """
        Tests if the multi gpu strategy can compile successfully on a multi gpu system.

        THIS TEST REQUIRES A MULTI GPU SYSTEM.
        """
        root_logger.setLevel(DEBUG)
        agent_config = config_from_path("configs/multi_gpu_ray_apex_for_pong.json")
        agent_config["execution_spec"].pop("ray_spec")
        environment = OpenAIGymEnv("Pong-v0", frameskip=4)

        agent = ApexAgent.from_spec(
            agent_config, state_space=environment.state_space, action_space=environment.action_space
        )
        print('Compiled apex agent')

    def test_apex_multi_gpu_update(self):
        """
        Tests if the multi GPU optimizer can perform successful updates, using the apex executor.
        """
        agent_config = config_from_path("configs/multi_gpu_ray_apex_for_pong.json")
        executor = ApexExecutor(
            environment_spec=self.env_spec,
            agent_config=agent_config,
        )

        # Executes actual workload.
        result = executor.execute_workload(workload=dict(
            num_timesteps=100000, report_interval=10000, report_interval_min_seconds=10)
        )

    def test_gpu_forward_pass_through_lstm_with_shared_variables(self):
        """
        Tests a forward pass through an LSTM on the GPU using distributed setup and shared variables on the CPU.
        """
        env_spec = dict(type="random", state_space=FloatBox(shape=(100,)), action_space=IntBox(9))
        dummy_process = multiprocessing.Process(target=dummy_func, args=[env_spec, self.cluster_spec])
        dummy_process.start()

        env = RandomEnv.from_spec(env_spec)
        state_space = env.state_space.with_extra_ranks(add_time_rank=True, add_batch_rank=True)

        lstm = LSTMLayer(units=50, time_major=True, device=dict(ops="/job:a/task:0/gpu", variables="/job:a/task:0/cpu"))
        test = ComponentTest(
            lstm,
            input_spaces=dict(inputs=state_space),
            action_space=env.action_space,
            execution_spec=dict(
                mode="distributed",
                gpu_spec=dict(
                    gpus_enabled=True,
                    max_usable_gpus=1,
                    num_gpus=1
                ),
                distributed_spec=dict(job="a", task_index=0, cluster_spec=self.cluster_spec),
                session_config=dict(
                    type="monitored-training-session",
                    allow_soft_placement=True,
                    log_device_placement=True
                ),
                enable_timeline=True,
            )
        )
        # Pass same sample n times through the LSTM.
        sample = state_space.sample()
        for _ in range(10):
            test.test("apply", sample, expected_outputs=None)

        dummy_process.join()

    # TODO (Bart maybe): We should probably have some tests that simply test the update call
    # This is just slightly annoying because we have to assemble a preprocessed batch manually
    # It would be good to have a utility method for that to use in tests (e.g. sample atari batches).