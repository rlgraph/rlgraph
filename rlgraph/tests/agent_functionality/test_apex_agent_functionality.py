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

import logging
import unittest
from rlgraph.agents import Agent
import rlgraph.spaces as spaces
from rlgraph.environments import RandomEnv
from rlgraph.tests import recursive_assert_almost_equal
from rlgraph.tests.test_util import config_from_path
from rlgraph.utils import root_logger


class TestApexAgentFunctionality(unittest.TestCase):
    """
    Tests Ape-X specific functionality.
    """
    root_logger.setLevel(level=logging.INFO)

    def test_apex_weight_syncing(self):
        env = RandomEnv(state_space=spaces.IntBox(2), action_space=spaces.IntBox(2), deterministic=True)

        agent = Agent.from_spec(
            config_from_path("configs/apex_agent_for_random_env.json"),
            state_space=env.state_space,
            action_space=env.action_space
        )

        policy_weights = agent.get_policy_weights()
        print('policy weights: {}'.format(policy_weights))

        for variable, weights in policy_weights.items():
            weights += 0.01
        agent.set_policy_weights(policy_weights)

        new_weights = agent.get_policy_weights()
        recursive_assert_almost_equal(policy_weights, new_weights)
