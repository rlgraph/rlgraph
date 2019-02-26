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

from rlgraph import get_backend
from rlgraph.agents import DQNAgent, ApexAgent, IMPALAAgent, ActorCriticAgent, PPOAgent, SACAgent
from rlgraph.environments import OpenAIGymEnv, GridWorld, GaussianDensityAsRewardEnvironment
from rlgraph.spaces import FloatBox, Tuple
from rlgraph.utils.util import default_dict
from rlgraph.tests.test_util import config_from_path


class TestAllCompile(unittest.TestCase):
    """
    Tests if all agents compile correctly on relevant configurations.
    """
    impala_cluster_spec = dict(learner=["localhost:22222"], actor=["localhost:22223"])

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
        print("Compiled {}".format(agent))

    def test_apex_compilation(self):
        """
        Tests agent compilation without Ray to ease debugging on Windows.
        """
        agent_config = config_from_path("configs/ray_apex_for_pong.json")
        agent_config["execution_spec"].pop("ray_spec")
        environment = OpenAIGymEnv("Pong-v0", frameskip=4)

        agent = ApexAgent.from_spec(
            agent_config, state_space=environment.state_space,
            action_space=environment.action_space
        )
        print("Compiled {}".format(agent))

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
        print("Compiled {}".format(agent))

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
        print("Compiled {}".format(agent))

    def test_impala_single_agent_compilation(self):
        """
        Tests IMPALA agent compilation (single-node mode).
        """
        return
        if get_backend() == "pytorch":
            return
        env = GridWorld("2x2")
        agent = IMPALAAgent.from_spec(
            config_from_path("configs/impala_agent_for_2x2_gridworld.json"),
            state_space=env.state_space,
            action_space=env.action_space,
            update_spec=dict(batch_size=16),
            optimizer_spec=dict(type="adam", learning_rate=0.05),
            # Make session-creation hang in docker.
            execution_spec=dict(disable_monitoring=True)
        )
        agent.terminate()
        print("Compiled {}".format(agent))

    def test_impala_actor_compilation(self):
        """
        Tests IMPALA agent compilation (actor).
        """
        return
        if get_backend() == "pytorch":
            return
        try:
            from rlgraph.environments.deepmind_lab import DeepmindLabEnv
        except ImportError:
            print("Deepmind Lab not installed: Will skip this test.")
            return

        agent_config = config_from_path("configs/impala_agent_for_deepmind_lab_env.json")
        env_spec = dict(level_id="seekavoid_arena_01", observations=["RGB_INTERLEAVED", "INSTR"], frameskip=4)
        dummy_env = DeepmindLabEnv.from_spec(env_spec)
        agent = IMPALAAgent.from_spec(
            agent_config,
            type="actor",
            state_space=dummy_env.state_space,
            action_space=dummy_env.action_space,
            internal_states_space=Tuple(FloatBox(shape=(256,)), FloatBox(shape=(256,)), add_batch_rank=False),
            environment_spec=default_dict(dict(type="deepmind-lab"), env_spec),
            # Make session-creation hang in docker.
            execution_spec=dict(
                session_config=dict(
                    type="monitored-training-session",
                    auto_start=False
                ),
                disable_monitoring=True
            )
        )
        # Start Specifiable Server with Env manually (monitoring is disabled).
        agent.environment_stepper.environment_server.start_server()
        print("Compiled {}".format(agent))
        agent.environment_stepper.environment_server.stop_server()
        agent.terminate()

    def test_impala_learner_compilation(self):
        """
        Tests IMPALA agent compilation (learner).
        """
        return
        if get_backend() == "pytorch":
            return
        try:
            from rlgraph.environments.deepmind_lab import DeepmindLabEnv
        except ImportError:
            print("Deepmind Lab not installed: Will skip this test.")
            return

        agent_config = config_from_path("configs/impala_agent_for_deepmind_lab_env.json")
        env_spec = dict(level_id="seekavoid_arena_01", observations=["RGB_INTERLEAVED", "INSTR"], frameskip=4)
        dummy_env = DeepmindLabEnv.from_spec(env_spec)
        learner_agent = IMPALAAgent.from_spec(
            agent_config,
            type="learner",
            state_space=dummy_env.state_space,
            action_space=dummy_env.action_space,
            internal_states_space=IMPALAAgent.default_internal_states_space,
            environment_spec=default_dict(dict(type="deepmind-lab"), env_spec),
            # Setup distributed tf.
            execution_spec=dict(
                mode="distributed",
                #gpu_spec=dict(
                #    gpus_enabled=True,
                #    max_usable_gpus=1,
                #    num_gpus=1
                #),
                distributed_spec=dict(job="learner", task_index=0, cluster_spec=self.impala_cluster_spec),
                session_config=dict(
                    type="monitored-training-session",
                    allow_soft_placement=True,
                    log_device_placement=True,
                    auto_start=False
                ),
                disable_monitoring=True,
                enable_timeline=True,
            )
        )
        print("Compiled IMPALA type=learner agent without starting the session (would block waiting for actor).")

        ## Take one batch from the filled up queue and run an update_from_memory with the learner.
        #update_steps = 10
        #time_start = time.perf_counter()
        #for _ in range(update_steps):
        #    agent.call_api_method("update_from_memory")
        #time_total = time.perf_counter() - time_start
        #print("Done learning {}xbatch-of-{} in {}sec ({} updates/sec).".format(
        #    update_steps, agent.update_spec["batch_size"], time_total , update_steps / time_total)
        #)

        learner_agent.terminate()

    def test_sac_compilation(self):
        # TODO: support SAC on pytorch.
        if get_backend() == "pytorch":
            return

        env = GaussianDensityAsRewardEnvironment(episode_length=5)
        agent = SACAgent.from_spec(
            config_from_path("configs/sac_agent_for_functionality_test.json"),
            state_space=env.state_space,
            action_space=env.action_space
        )
        print("Compiled {}".format(agent))
