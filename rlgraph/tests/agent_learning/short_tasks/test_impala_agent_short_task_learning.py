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

import numpy as np
import unittest
import logging

from rlgraph.environments import GridWorld
from rlgraph.agents import IMPALAAgent
from rlgraph.utils import root_logger
from rlgraph.tests.test_util import config_from_path, recursive_assert_almost_equal


class TestIMPALAAgentShortTaskLearning(unittest.TestCase):
    """
    Tests whether the DQNAgent can learn in simple environments.
    """
    root_logger.setLevel(level=logging.INFO)

    def test_impala_on_2x2_grid_world(self):
        """
        Creates a single IMPALAAgent and runs it via the IMPALAWorker on a simple 2x2 GridWorld.
        """
        env = GridWorld("2x2")
        agent = IMPALAAgent.from_spec(
            config_from_path("configs/impala_agent_for_2x2_gridworld.json"),
            state_space=env.state_space,
            action_space=env.action_space,
            execution_spec=dict(seed=12),
            update_spec=dict(update_interval=4, batch_size=16),
            optimizer_spec=dict(type="adam", learning_rate=0.05),
        )

        learn_updates = 1000
        # Setup the queue runner.
        agent.call_api_method("setup_queue_runner")
        for _ in range(learn_updates):
            agent.update()


        #print("STATES:\n{}".format(agent.last_q_table["states"]))
        #print("\n\nQ(s,a)-VALUES:\n{}".format(np.round_(agent.last_q_table["q_values"], decimals=2)))

        #self.assertEqual(results["timesteps_executed"], time_steps)
        #self.assertEqual(results["env_frames"], time_steps)
        #self.assertGreaterEqual(results["mean_episode_reward"], -3.5)
        #self.assertGreaterEqual(results["max_episode_reward"], 0.0)
        #self.assertLessEqual(results["episodes_executed"], 350)

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

