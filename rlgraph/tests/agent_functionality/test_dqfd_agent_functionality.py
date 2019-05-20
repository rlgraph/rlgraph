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

import numpy as np

from rlgraph.agents import DQFDAgent
from rlgraph.environments import OpenAIGymEnv
from rlgraph.spaces import BoolBox, FloatBox, IntBox, Dict
from rlgraph.tests.test_util import config_from_path, recursive_assert_almost_equal


class TestDQFDAgentFunctionality(unittest.TestCase):
    """
    Tests the DQFD Agent's functionality.
    """
    env_spec = dict(type="openai", gym_env="CartPole-v0")

    def test_container_actions(self):
        # Test container actions with embedding.

        vocab_size = 100
        embed_dim = 128
        # ID/state space.
        state_space = IntBox(vocab_size, shape=(10,))
        # Container action space.
        actions_space = {}
        num_outputs = 3
        for i in range(3):
            actions_space['action_{}'.format(i)] = IntBox(
                low=0,
                high=num_outputs
            )
        actions_space = Dict(actions_space)

        agent_config = config_from_path("configs/dqfd_container.json")
        agent_config["network_spec"] = [
                dict(type="embedding", embed_dim=embed_dim, vocab_size=vocab_size),
                dict(type="reshape", flatten=True),
                dict(type="dense", units=embed_dim, activation="relu", scope="dense_1")
            ]
        agent = DQFDAgent.from_spec(
            agent_config,
            state_space=state_space,
            action_space=actions_space
        )
        terminals = BoolBox(add_batch_rank=True)
        rewards = FloatBox(add_batch_rank=True)

        agent.observe_demos(
            preprocessed_states=agent.preprocessed_state_space.with_batch_rank().sample(1),
            actions=actions_space.with_batch_rank().sample(1),
            rewards=rewards.sample(1),
            next_states=agent.preprocessed_state_space.with_batch_rank().sample(1),
            terminals=terminals.sample(1),
        )

    def test_insert_demos(self):
        """
        Tests inserting into the demo memory.
        """
        env = OpenAIGymEnv.from_spec(self.env_spec)

        agent_config = config_from_path("configs/dqfd_agent_for_cartpole.json")
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
        env = OpenAIGymEnv.from_spec(self.env_spec)
        agent_config = config_from_path("configs/dqfd_agent_for_cartpole.json")
        agent = DQFDAgent.from_spec(
            agent_config,
            state_space=env.state_space,
            action_space=env.action_space
        )
        terminals = BoolBox(add_batch_rank=True)
        rewards = FloatBox(add_batch_rank=True)
        state_1 = agent.preprocessed_state_space.with_batch_rank().sample(1)
        action_1 = [1]
        state_2 = agent.preprocessed_state_space.with_batch_rank().sample(1)
        action_2 = [0]

        # Insert two states with fixed actions and a few random examples.
        for _ in range(10):
            # State with correct action
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

        # Update.
        agent.update_from_demos(batch_size=8, num_updates=1000)

        # Test if fixed states and actions map.
        action = agent.get_action(states=state_1, apply_preprocessing=False, use_exploration=False)
        self.assertEqual(action, action_1)

        action = agent.get_action(states=state_2, apply_preprocessing=False, use_exploration=False)
        self.assertEqual(action, action_2)

    def test_demos_with_container_actions(self):
        # Tests if dqfd can fit a set of states to a set of actions.
        vocab_size = 100
        embed_dim = 128
        # ID/state space.
        state_space = IntBox(vocab_size, shape=(10,))
        # Container action space.
        actions_space = {}
        num_outputs = 3
        for i in range(3):
            actions_space['action_{}'.format(i)] = IntBox(
                low=0,
                high=num_outputs
            )
        actions_space = Dict(actions_space)

        agent_config = config_from_path("configs/dqfd_container.json")
        agent_config["network_spec"] = [
                dict(type="embedding", embed_dim=embed_dim, vocab_size=vocab_size),
                dict(type="reshape", flatten=True),
                dict(type="dense", units=embed_dim, activation="relu", scope="dense_1")
            ]
        agent = DQFDAgent.from_spec(
            agent_config,
            state_space=state_space,
            action_space=actions_space
        )
        terminals = BoolBox(add_batch_rank=True)
        rewards = FloatBox(add_batch_rank=True)

        # Create a set of demos.
        demo_states = agent.preprocessed_state_space.with_batch_rank().sample(20)
        demo_actions = actions_space.with_batch_rank().sample(20)
        demo_rewards = rewards.sample(20, fill_value=1.0)
        demo_next_states = agent.preprocessed_state_space.with_batch_rank().sample(20)
        demo_terminals = terminals.sample(20, fill_value=False)

        # Insert.
        agent.observe_demos(
            preprocessed_states=demo_states,
            actions=demo_actions,
            rewards=demo_rewards,
            next_states=demo_next_states,
            terminals=demo_terminals,
        )

        # Fit demos.
        agent.update_from_demos(batch_size=20, num_updates=5000)

        # Evaluate demos:
        agent_actions = agent.get_action(demo_states, apply_preprocessing=False, use_exploration=False)
        recursive_assert_almost_equal(agent_actions, demo_actions)

    def test_update_online(self):
        """
        Tests if joint updates from demo and online memory work.
        """
        env = OpenAIGymEnv.from_spec(self.env_spec)
        agent_config = config_from_path("configs/dqfd_agent_for_cartpole.json")
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
            internals=[],
            terminals=terminals.sample(32),
            next_states=agent.preprocessed_state_space.sample(32)
        )
        # Call update.
        agent.update()

    def test_custom_margin_demos_with_container_actions(self):
        # Tests if using different margins per sample works.
        # Same state, but different
        vocab_size = 100
        embed_dim = 8
        # ID/state space.
        state_space = IntBox(vocab_size, shape=(10,))
        # Container action space.
        actions_space = {}
        num_outputs = 3
        for i in range(3):
            actions_space['action_{}'.format(i)] = IntBox(
                low=0,
                high=num_outputs
            )
        actions_space = Dict(actions_space)

        agent_config = config_from_path("configs/dqfd_container.json")
        agent_config["network_spec"] = [
            dict(type="embedding", embed_dim=embed_dim, vocab_size=vocab_size),
            dict(type="reshape", flatten=True),
            dict(type="dense", units=embed_dim, activation="relu", scope="dense_1")
        ]
        agent = DQFDAgent.from_spec(
            agent_config,
            state_space=state_space,
            action_space=actions_space
        )
        terminals = BoolBox(add_batch_rank=True)
        rewards = FloatBox(add_batch_rank=True)

        # Create a set of demos.
        demo_states = agent.preprocessed_state_space.with_batch_rank().sample(2)
        # Same state.
        demo_states[1] = demo_states[0]
        demo_actions = actions_space.with_batch_rank().sample(2)

        for name, action in actions_space.items():
            demo_actions[name][0] = 0
            demo_actions[name][1] = 1

        demo_rewards = rewards.sample(2, fill_value=.0)
        # One action has positive reward, one negative
        demo_rewards[0] = 0
        demo_rewards[1] = 0

        # One action is encouraged, one is discouraged.
        margins = np.asarray([0.5, -0.5])

        demo_next_states = agent.preprocessed_state_space.with_batch_rank().sample(2)
        demo_terminals = terminals.sample(2, fill_value=False)

        # When using margins, need to use external batch.
        batch = dict(
            states=demo_states,
            actions=demo_actions,
            rewards=demo_rewards,
            next_states=demo_next_states,
            importance_weights=np.ones_like(demo_rewards),
            terminals=demo_terminals,
        )
        # Fit demos with custom margins.
        for _ in range(10000):
            agent.update(batch=batch, update_from_demos=False, apply_demo_loss_to_batch=True, expert_margins=margins)

        # Evaluate demos for the state -> should have action with positive reward.
        agent_actions = agent.get_action(np.array([demo_states[0]]), apply_preprocessing=False, use_exploration=False)
        print("learned action = ", agent_actions)
