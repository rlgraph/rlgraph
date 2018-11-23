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

from rlgraph import get_backend
from rlgraph.agents import DQNAgent, ApexAgent, IMPALAAgent, ActorCriticAgent, PPOAgent
from rlgraph.environments import OpenAIGymEnv, GridWorld
from rlgraph.spaces import FloatBox, Tuple
from rlgraph.tests.test_util import config_from_path


class TestAllCompile(unittest.TestCase):
    """
    Tests if all agents compile correctly on relevant configurations.
    """
    def test_dqn_compilation(self):
        """
        Tests DQN Agent compilation.
        """
        env = OpenAIGymEnv("Pong-v0", frameskip=4, max_num_noops=30, episodic_life=True)
        agent_config = config_from_path("configs/dqn_agent_for_pong.json")
        agent = DQNAgent.from_spec(
            # Uses 2015 DQN parameters as closely as possible.
            agent_config,
            state_space=env.state_space,
            # Try with "reduced" action space (actually only 3 actions, up, down, no-op)
            action_space=env.action_space
        )

    def test_apex_compilation(self):
        """
        Tests agent compilation without Ray to ease debugging on Windows.
        """
        agent_config = config_from_path("configs/ray_apex_for_pong.json")
        agent_config["execution_spec"].pop("ray_spec")
        # TODO remove after unified.
        if get_backend() == "pytorch":
            agent_config["memory_spec"]["type"] = "mem_prioritized_replay"
        environment = OpenAIGymEnv("Pong-v0", frameskip=4)

        agent = ApexAgent.from_spec(
            agent_config, state_space=environment.state_space,
            action_space=environment.action_space
        )
        print('Compiled apex agent')

    def test_actor_critic_compilation(self):
        """
        Tests Policy gradient agent compilation.
        """
        env = OpenAIGymEnv("Pong-v0", frameskip=4, max_num_noops=30, episodic_life=True)
        agent_config = config_from_path("configs/actor_critic_agent_for_pong.json")
        agent = ActorCriticAgent.from_spec(
            agent_config,
            state_space=env.state_space,
            action_space=env.action_space
        )

    def test_ppo_compilation(self):
        """
        Tests PPO agent compilation.
        """
        env = OpenAIGymEnv("Pong-v0", frameskip=4, max_num_noops=30, episodic_life=True)
        agent_config = config_from_path("configs/ppo_agent_for_pong.json")
        agent = PPOAgent.from_spec(
            agent_config,
            state_space=env.state_space,
            action_space=env.action_space
        )

    #def test_impala_single_agent_compilation(self):
    #    """
    #    Tests IMPALA agent compilation (single-node mode).
    #    """
    #    env = GridWorld("2x2")
    #    agent = IMPALAAgent.from_spec(
    #        config_from_path("configs/impala_agent_for_2x2_gridworld.json"),
    #        state_space=env.state_space,
    #        action_space=env.action_space,
    #        execution_spec=dict(seed=12),
    #        update_spec=dict(batch_size=16),
    #        optimizer_spec=dict(type="adam", learning_rate=0.05),
    #        batch_apply=True
    #    )
    #    agent.terminate()
    #    print("Compiled IMPALA type=actor agent.")

    def test_impala_actor_compilation(self):
        """
        Tests IMPALA agent compilation (actor).
        """
        try:
            from rlgraph.environments.deepmind_lab import DeepmindLabEnv
        except ImportError:
            print("Deepmind Lab not installed: Will skip this test.")
            return

        agent_config = config_from_path("configs/impala_agent_for_deepmind_lab_env.json")
        env = DeepmindLabEnv(
            level_id="seekavoid_arena_01", observations=["RGB_INTERLEAVED", "INSTR"], frameskip=4
        )

        actor_agent = IMPALAAgent.from_spec(
            agent_config,
            type="actor",
            state_space=env.state_space,
            action_space=env.action_space,
            internal_states_space=Tuple(FloatBox(shape=(256,)), FloatBox(shape=(256,)), add_batch_rank=True),
            # Make session-creation hang in docker.
            execution_spec=dict(disable_monitoring=True)
        )
        # Start Specifiable Server with Env manually.
        actor_agent.environment_stepper.environment_server.start()
        print("Compiled IMPALA type=actor agent.")
        actor_agent.environment_stepper.environment_server.stop()

    def test_impala_learner_compilation(self):
        """
        Tests IMPALA agent compilation (learner).
        """
        try:
            from rlgraph.environments.deepmind_lab import DeepmindLabEnv
        except ImportError:
            print("Deepmind Lab not installed: Will skip this test.")
            return

        agent_config = config_from_path("configs/impala_agent_for_deepmind_lab_env.json")
        env = DeepmindLabEnv(
            level_id="seekavoid_arena_01", observations=["RGB_INTERLEAVED", "INSTR"], frameskip=4
        )

        agent = IMPALAAgent.from_spec(
            agent_config,
            type="learner",
            state_space=env.state_space,
            action_space=env.action_space,
            internal_states_space=Tuple(FloatBox(shape=(256,)), FloatBox(shape=(256,)), add_batch_rank=True),
        )
        agent.terminate()

        print("Compiled IMPALA type=learner agent.")
