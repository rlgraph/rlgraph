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

from rlgraph.agents import DQFDAgent
from rlgraph.environments import OpenAIGymEnv
from rlgraph.spaces import BoolBox, FloatBox
from rlgraph.tests.test_util import config_from_path


class TestDQNAgentFunctionality(unittest.TestCase):
    """
    Tests the DQFD Agent's functionality.
    """
    def test_insert_demos(self):
        """
        Tests inserting into the demo memory.
        """
        env_spec = dict(
            type="openai",
            gym_env="PongNoFrameskip-v4"
        )
        env = OpenAIGymEnv.from_spec(env_spec)

        agent_config = config_from_path("configs/dqfd_agent_for_pong.json")
        agent = DQFDAgent.from_spec(
            agent_config,
            state_space=env.state_space,
            action_space=env.action_space
        )
        terminals = BoolBox(add_batch_rank=True)
        rewards = FloatBox(add_batch_rank=True)

        # Observe a single data point.
        agent.observe_demos(
            preprocessed_states=agent.preprocessed_state_space.with_batch_rank().sample(1),
            actions=env.action_space.with_batch_rank().sample(1),
            rewards=rewards.sample(1),
            next_states=agent.preprocessed_state_space.with_batch_rank().sample(1),
            terminals=terminals.sample(1),
        )

        # Observe a batch of demos.
        agent.observe_demos(
            preprocessed_states=agent.preprocessed_state_space.sample(10),
            actions=env.action_space.sample(10),
            rewards=FloatBox().sample(10),
            terminals=terminals.sample(10),
            next_states=agent.preprocessed_state_space.sample(10)
        )

    def test_update_from_demos(self):
        """
        Tests the separate API method to update from demos.
        """
        env_spec = dict(
            type="openai",
            gym_env="PongNoFrameskip-v4"
        )
        env = OpenAIGymEnv.from_spec(env_spec)
        # TODO use smaller network/env for this test.
        agent_config = config_from_path("configs/dqfd_agent_for_pong.json")
        agent = DQFDAgent.from_spec(
            agent_config,
            state_space=env.state_space,
            action_space=env.action_space
        )
        terminals = BoolBox(add_batch_rank=True)
        rewards = FloatBox(add_batch_rank=True)
        state_1 = agent.preprocessed_state_space.with_batch_rank().sample(1)
        action_1 = [2]
        state_2 = agent.preprocessed_state_space.with_batch_rank().sample(1)
        action_2 = [0]

        # Insert two states with fixed actions and a few random examples.
        for _ in range(10):
            agent.observe_demos(
                preprocessed_states=state_1,
                actions=action_1,
                rewards=rewards.sample(1),
                next_states=agent.preprocessed_state_space.with_batch_rank().sample(1),
                terminals=terminals.sample(1),
            )
            agent.observe_demos(
                preprocessed_states=state_2,
                actions=action_2,
                rewards=rewards.sample(1),
                next_states=agent.preprocessed_state_space.with_batch_rank().sample(1),
                terminals=terminals.sample(1),
            )
            agent.observe_demos(
                preprocessed_states=agent.preprocessed_state_space.sample(2),
                actions=env.action_space.sample(2),
                rewards=FloatBox().sample(2),
                terminals=terminals.sample(2),
                next_states=agent.preprocessed_state_space.sample(2)
            )

        # Update.
        agent.update_from_demos(num_updates=100, batch_size=16)

        # Test if fixed states and actions map.
        action = agent.get_action(states=state_1, apply_preprocessing=False)
        self.assertEqual(action, action_1)

        action = agent.get_action(states=state_2, apply_preprocessing=False)
        self.assertEqual(action, action_2)

    def test_update_online(self):
        """
        Tests if joint updates from demo and online memory work.
        """
        env_spec = dict(
            type="openai",
            gym_env="PongNoFrameskip-v4"
        )
        env = OpenAIGymEnv.from_spec(env_spec)

        agent_config = config_from_path("configs/dqfd_agent_for_pong.json")
        agent = DQFDAgent.from_spec(
            agent_config,
            state_space=env.state_space,
            action_space=env.action_space
        )
        terminals = BoolBox(add_batch_rank=True)

        # Observe a batch of demos.
        agent.observe_demos(
            preprocessed_states=agent.preprocessed_state_space.sample(32),
            actions=env.action_space.sample(32),
            rewards=FloatBox().sample(32),
            terminals=terminals.sample(32),
            next_states=agent.preprocessed_state_space.sample(32)
        )

        # Observe a batch of online data.
        agent._observe_graph(
            preprocessed_states=agent.preprocessed_state_space.sample(32),
            actions=env.action_space.sample(32),
            rewards=FloatBox().sample(32),
            terminals=terminals.sample(32),
            next_states=agent.preprocessed_state_space.sample(32)
        )
        # Call update.
        agent.update()
