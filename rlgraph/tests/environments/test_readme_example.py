# Copyright 2018/2019 The RLgraph authors. All Rights Reserved.
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

from rlgraph.tests.test_util import config_from_path


class TestReadmeExample(unittest.TestCase):
    """
    Tests if the readme example runs.
    """

    def test_readme_example(self):
        """
        Tests deterministic functionality of RandomEnv.
        """
        from rlgraph.agents import DQNAgent
        from rlgraph.environments import OpenAIGymEnv

        environment = OpenAIGymEnv('CartPole-v0')
        config = config_from_path("../../examples/configs/dqn_cartpole.json")

        # Create from .json file or dict, see agent API for all
        # possible configuration parameters.
        agent = DQNAgent.from_spec(
            config,
            state_space=environment.state_space,
            action_space=environment.action_space
        )

        # Get an action, take a step, observe reward.
        state = environment.reset()
        action, preprocessed_state = agent.get_action(
            states=state,
            extra_returns="preprocessed_states"
        )

        # Execute step in environment.
        next_state, reward, terminal, info = environment.step(action)

        # Observe result.
        agent.observe(
            preprocessed_states=preprocessed_state,
            actions=action,
            internals=[],
            next_states=next_state,
            rewards=reward,
            terminals=terminal
        )

        # Call update when desired:
        loss = agent.update()
