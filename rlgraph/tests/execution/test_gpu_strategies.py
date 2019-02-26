# Copyright 2018/2019 The RLgraph authors. All Rights Reserved.
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

import numpy as np
import unittest
from rlgraph.utils import root_logger
from logging import DEBUG

from rlgraph.agents import ApexAgent, DQNAgent, PPOAgent
from rlgraph.environments import OpenAIGymEnv, RandomEnv, GridWorld
#from rlgraph.execution.ray import ApexExecutor
from rlgraph.spaces import *
from rlgraph.tests.test_util import config_from_path
from rlgraph.execution.single_threaded_worker import SingleThreadedWorker
from rlgraph.tests.test_util import recursive_assert_almost_equal


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
    random_env_spec = dict(type="random", state_space=FloatBox(shape=(2,)), action_space=IntBox(2))
    grid_world_2x2_flattened_state_space = FloatBox(shape=(4,), add_batch_rank=True)
    grid_world_4x4_flattened_state_space = FloatBox(shape=(16,), add_batch_rank=True)

    def test_multi_gpu_dqn_agent_compilation(self):
        """
        Tests if the multi gpu strategy can compile successfully on a multi gpu system, but
        also runs on a CPU-only system using fake-GPU logic for testing purposes.
        """
        root_logger.setLevel(DEBUG)
        agent_config = config_from_path("configs/multi_gpu_dqn_for_random_env.json")
        environment = RandomEnv.from_spec(self.random_env_spec)

        agent = DQNAgent.from_spec(
            agent_config, state_space=environment.state_space, action_space=environment.action_space
        )
        print("Compiled DQN agent on multi-GPU system")

        # Do an update from external batch.
        batch_size = agent_config["update_spec"]["batch_size"]
        external_batch = dict(
            states=environment.state_space.sample(size=batch_size),
            actions=environment.action_space.sample(size=batch_size),
            rewards=np.random.sample(size=batch_size),
            terminals=np.random.choice([True, False], size=batch_size),
            next_states=environment.state_space.sample(size=batch_size),
            importance_weights=np.zeros(shape=(batch_size,))
        )
        agent.update(batch=external_batch)
        print("Performed an update from external batch")

    def test_multi_gpu_apex_agent_compilation(self):
        """
        Tests if the multi gpu strategy can compile successfully on a multi gpu system, but
        also runs on a CPU-only system using fake-GPU logic for testing purposes.
        """
        root_logger.setLevel(DEBUG)
        agent_config = config_from_path("configs/multi_gpu_ray_apex_for_pong.json")
        agent_config["execution_spec"].pop("ray_spec")
        environment = OpenAIGymEnv("Pong-v0", frameskip=4)

        agent = ApexAgent.from_spec(
            agent_config, state_space=environment.state_space, action_space=environment.action_space
        )
        print("Compiled Apex agent")

    def test_multi_gpu_dqn_agent_learning_test_gridworld_2x2(self):
        """
        Tests if the multi gpu strategy can learn successfully on a multi gpu system, but
        also runs on a CPU-only system using fake-GPU logic for testing purposes.
        """
        env_spec = dict(type="grid-world", world="2x2")
        dummy_env = GridWorld.from_spec(env_spec)
        agent_config = config_from_path("configs/multi_gpu_dqn_for_2x2_gridworld.json")
        preprocessing_spec = agent_config.pop("preprocessing_spec")
        agent = DQNAgent.from_spec(
            agent_config,
            state_space=self.grid_world_2x2_flattened_state_space,
            action_space=dummy_env.action_space,
        )

        time_steps = 1000
        worker = SingleThreadedWorker(
            env_spec=env_spec,
            agent=agent,
            worker_executes_preprocessing=True,
            preprocessing_spec=preprocessing_spec
        )
        results = worker.execute_timesteps(time_steps, use_exploration=True)

        # Marge q-tables of all four GPUs:
        agent.last_q_table["q_values"] = agent.last_q_table["q_values"].reshape((48, 4))

        print("STATES:\n{}".format(agent.last_q_table["states"]))
        print("\n\nQ(s,a)-VALUES:\n{}".format(np.round_(agent.last_q_table["q_values"], decimals=2)))

        self.assertEqual(results["timesteps_executed"], time_steps)
        self.assertEqual(results["env_frames"], time_steps)
        self.assertGreaterEqual(results["mean_episode_reward"], -4.5)
        self.assertGreaterEqual(results["max_episode_reward"], 0.0)
        self.assertLessEqual(results["episodes_executed"], time_steps / 2)

        # Check q-table for correct values.
        expected_q_values_per_state = {
            (1.0, 0, 0, 0): (-1, -5, 0, -1),
            (0, 1.0, 0, 0): (-1, 1, 0, 0)
        }
        for state, q_values in zip(agent.last_q_table["states"], agent.last_q_table["q_values"]):
            state, q_values = tuple(state), tuple(q_values)
            assert state in expected_q_values_per_state, \
                "ERROR: state '{}' not expected in q-table as it's a terminal state!".format(state)
            recursive_assert_almost_equal(q_values, expected_q_values_per_state[state], decimals=0)

    def test_apex_multi_gpu_update(self):
        """
        Tests if the multi GPU optimizer can perform successful updates, using the apex executor.
        Also runs on a CPU-only system using fake-GPU logic for testing purposes.
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

    def test_multi_gpu_ppo_agent_learning_test_gridworld_2x2(self):
        """
        Tests if the multi gpu strategy can learn successfully on a multi gpu system, but
        also runs on a CPU-only system using fake-GPU logic for testing purposes.
        """
        env_spec = dict(type="grid-world", world="2x2")
        dummy_env = GridWorld.from_spec(env_spec)
        agent_config = config_from_path("configs/multi_gpu_ppo_for_2x2_gridworld.json")
        preprocessing_spec = agent_config.pop("preprocessing_spec")
        agent = PPOAgent.from_spec(
            agent_config,
            state_space=self.grid_world_2x2_flattened_state_space,
            action_space=dummy_env.action_space,
        )

        time_steps = 10000
        worker = SingleThreadedWorker(
            env_spec=env_spec,
            agent=agent,
            worker_executes_preprocessing=True,
            preprocessing_spec=preprocessing_spec
        )
        results = worker.execute_timesteps(time_steps, use_exploration=True)

        # Assume we have learned something.
        # TODO: This test needs more tuning. -1.0 is not great for the 2x2 grid world.
        self.assertGreater(results["mean_episode_reward"], -1.0)
