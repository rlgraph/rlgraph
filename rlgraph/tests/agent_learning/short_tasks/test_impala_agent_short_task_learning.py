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
            update_spec=dict(batch_size=16),
            optimizer_spec=dict(type="adam", learning_rate=0.005),
        )

        learn_updates = 300
        # Setup the queue runner.
        agent.call_api_method("setup_queue_runner")
        for i in range(learn_updates):
            ret = agent.update()
            print("{}".format(i))
            # Calculate return per episode.
            rewards = ret[3]["rewards"][:, 1:].reshape((80,))
            terminals = ret[3]["terminals"][:, 1:].reshape((80,))
            returns = list()
            return_ = 0.0
            for r, t in zip(rewards, terminals):
                return_ += r
                if t:
                    returns.append(return_)
                    return_ = 0.0

            print("\tLoss={:.2} Avg-reward={}".format(float(ret[1]), np.mean(returns)))

        # Check the last action probs for the 2 valid next_states (start (after a reset) and one below start).
        action_probs = ret[3]["action_probs"][:, 1:, :].reshape((80, 4))
        next_states = ret[3]["states"][:, 1:].reshape((80,))
        for s_, probs in zip(next_states, action_probs):
            # Start state: Assume we picked "right" in state=1 (in order to step into goal state).
            if s_ == 0:
                recursive_assert_almost_equal(probs[0], 0.0, decimals=2)
                recursive_assert_almost_equal(probs[1], 0.99, decimals=2)
                recursive_assert_almost_equal(probs[2], 0.0, decimals=2)
                recursive_assert_almost_equal(probs[3], 0.0, decimals=2)
            # One below start: Assume we picked "down" in start state with very large probability.
            elif s_ == 1:
                recursive_assert_almost_equal(probs[0], 0.0, decimals=2)
                recursive_assert_almost_equal(probs[1], 0.0, decimals=2)
                recursive_assert_almost_equal(probs[2], 0.99, decimals=2)
                recursive_assert_almost_equal(probs[3], 0.0, decimals=2)

        agent.terminate()

