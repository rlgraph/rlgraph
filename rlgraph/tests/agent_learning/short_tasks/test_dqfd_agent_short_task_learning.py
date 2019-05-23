# Copyright 2018/2019 ducandu GmbH. All Rights Reserved.
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

import logging
import os
import unittest

import numpy as np

from rlgraph.agents import DQFDAgent
from rlgraph.environments import GridWorld
from rlgraph.execution import SingleThreadedWorker
from rlgraph.spaces import FloatBox
from rlgraph.tests.test_util import config_from_path, recursive_assert_almost_equal
from rlgraph.utils import root_logger
from rlgraph.utils.numpy import one_hot


class TestDQNAgentShortTaskLearning(unittest.TestCase):
    """
    Tests whether the DQNAgent can learn in simple environments.
    """
    root_logger.setLevel(level=logging.INFO)
    # Preprocessed state spaces.
    grid_world_2x2_flattened_state_space = FloatBox(shape=(4,), add_batch_rank=True)
    grid_world_4x4_flattened_state_space = FloatBox(shape=(16,), add_batch_rank=True)
    is_windows = os.name == "nt"

    def test_dqfd_on_2x2_grid_world(self):
        """
        Creates a DQNAgent and runs it via a Runner on a simple 2x2 GridWorld.
        """
        dummy_env = GridWorld("2x2")
        agent_config = config_from_path("configs/dqn_agent_for_2x2_gridworld.json")
        preprocessing_spec = agent_config.pop("preprocessing_spec")
        agent = DQFDAgent.from_spec(
            agent_config,
            double_q=False,
            dueling_q=False,
            state_space=self.grid_world_2x2_flattened_state_space,
            action_space=dummy_env.action_space,
            execution_spec=dict(seed=12),
            memory_batch_size=24,
            sync_every_n_updates=8,
            optimizer_spec=dict(type="adam", learning_rate=0.05)
        )

        time_steps = 2000
        worker = SingleThreadedWorker(
            env_spec=lambda: GridWorld("2x2"),
            agent=agent,
            preprocessing_spec=preprocessing_spec,
            worker_executes_preprocessing=True,
            update_rules=dict(update_every_n_units=4)
        )
        results = worker.execute_timesteps(time_steps, use_exploration=True)

        self.assertEqual(results["timesteps_executed"], time_steps)
        self.assertEqual(results["env_frames"], time_steps)
        self.assertGreaterEqual(results["mean_episode_reward"], -1.5)
        self.assertGreaterEqual(results["max_episode_reward"], 0.0)
        self.assertLessEqual(results["episodes_executed"], time_steps / 2)

        # Check all learnt Q-values.
        q_values = agent.graph_executor.execute(("get_q_values", one_hot(np.array([0, 1]), depth=4)))[:]
        recursive_assert_almost_equal(q_values[0], (0.8, -5, 0.9, 0.8), decimals=1)
        recursive_assert_almost_equal(q_values[1], (0.8, 1.0, 0.9, 0.9), decimals=1)
