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

import logging
import os
import unittest

import numpy as np
from scipy import stats

from rlgraph.agents.sac_agent import SACAgentComponent, SACAgent, SyncSpecification
from rlgraph.components import Policy, ValueFunction, PreprocessorStack, ReplayMemory, AdamOptimizer, \
    Synchronizable
from rlgraph.environments import GaussianDensityAsRewardEnv, OpenAIGymEnv, GridWorld
from rlgraph.execution import SingleThreadedWorker
from rlgraph.spaces import FloatBox, BoolBox
from rlgraph.tests import ComponentTest
from rlgraph.tests.test_util import config_from_path
from rlgraph.utils import root_logger


class TestSACShortTaskLearning(unittest.TestCase):
    """
    Tests whether the SACAgent and the SACAgentComponent can learn in simple environments.
    """
    root_logger.setLevel(level=logging.INFO)

    is_windows = os.name == "nt"

    def test_sac_agent_component_on_fake_env(self):
        config = config_from_path("configs/sac_component_for_fake_env_test.json")

        # Arbitrary state space, state should not be used in this example.
        state_space = FloatBox(shape=(2,))
        continuous_action_space = FloatBox(low=-1.0, high=1.0)
        terminal_space = BoolBox(add_batch_rank=True)
        policy = Policy.from_spec(config["policy"], action_space=continuous_action_space)
        policy.add_components(Synchronizable(), expose_apis="sync")
        q_function = ValueFunction.from_spec(config["value_function"])

        agent_component = SACAgentComponent(
            agent=None,
            policy=policy,
            q_function=q_function,
            preprocessor=PreprocessorStack.from_spec([]),
            memory=ReplayMemory.from_spec(config["memory"]),
            discount=config["discount"],
            initial_alpha=config["initial_alpha"],
            target_entropy=None,
            optimizer=AdamOptimizer.from_spec(config["optimizer"]),
            vf_optimizer=AdamOptimizer.from_spec(config["value_function_optimizer"], scope="vf-optimizer"),
            alpha_optimizer=None,
            q_sync_spec=SyncSpecification(sync_interval=10, sync_tau=1.0),
            num_q_functions=2
        )

        test = ComponentTest(
            component=agent_component,
            input_spaces=dict(
                states=state_space.with_batch_rank(),
                preprocessed_states=state_space.with_batch_rank(),
                actions=continuous_action_space.with_batch_rank(),
                rewards=FloatBox(add_batch_rank=True),
                next_states=state_space.with_batch_rank(),
                terminals=terminal_space,
                batch_size=int,
                preprocessed_s_prime=state_space.with_batch_rank(),
                importance_weights=FloatBox(add_batch_rank=True),
                preprocessed_next_states=state_space.with_batch_rank(),
                deterministic=bool,
                weights="variables:{}".format(policy.scope),
                # TODO: how to provide the space for multiple component variables?
                # q_weights=Dict(
                #    q_0="variables:{}".format(q_function.scope),
                #    q_1="variables:{}".format(agent_component._q_functions[1].scope),
                # )
            ),
            action_space=continuous_action_space,
            build_kwargs=dict(
                optimizer=agent_component._optimizer,
                build_options=dict(
                    vf_optimizer=agent_component.vf_optimizer,
                ),
            )
        )

        policy_loss = []
        vf_loss = []

        # This test simulates an env that always requires actions to be close to the max-pdf
        # value of a loc=0.5, scale=0.2 normal, regardless of any state inputs.
        # The component should learn to produce actions like that (close to 0.5).
        true_mean = 0.5
        target_dist = stats.norm(loc=true_mean, scale=0.2)
        batch_size = 100
        for _ in range(5000):
            action_sample = continuous_action_space.sample(batch_size)
            rewards = target_dist.pdf(action_sample)
            result = test.test(("update_from_external_batch", [
                state_space.sample(batch_size),
                action_sample,
                rewards,
                [True] * batch_size,
                state_space.sample(batch_size),
                [1.0] * batch_size  # importance
            ]))
            policy_loss.append(result["actor_loss"])
            vf_loss.append(result["critic_loss"])

        self.assertTrue(np.mean(policy_loss[:100]) > np.mean(policy_loss[-100:]))
        self.assertTrue(np.mean(vf_loss[:100]) > np.mean(vf_loss[-100:]))

        action_sample = np.linspace(-1, 1, batch_size)
        q_values = test.test(("get_q_values", [state_space.sample(batch_size), action_sample]))
        for q_val in q_values:
            q_val = q_val.flatten()
            np.testing.assert_allclose(q_val, target_dist.pdf(action_sample), atol=0.2)

        action_sample, _ = test.test(("action_from_preprocessed_state", [state_space.sample(batch_size), False]))
        action_sample = action_sample.flatten()
        np.testing.assert_allclose(np.mean(action_sample), true_mean, atol=0.1)

    def test_sac_learning_on_gaussian_density_as_reward_env(self):
        """
        Creates an SAC-Agent and runs it via a Runner on the GaussianDensityAsRewardEnv.
        """
        env = GaussianDensityAsRewardEnv(episode_length=5)
        agent = SACAgent.from_spec(
            config_from_path("configs/sac_agent_for_gaussian_density_env.json"),
            state_space=env.state_space,
            action_space=env.action_space
        )

        worker = SingleThreadedWorker(env_spec=lambda: env, agent=agent)
        worker.execute_episodes(num_episodes=500)
        rewards = worker.finished_episode_rewards[0]  # 0=1st env in vector-env
        self.assertTrue(np.mean(rewards[:100]) < np.mean(rewards[-100:]))

        worker.execute_episodes(num_episodes=100, use_exploration=False, update_spec=None)
        rewards = worker.finished_episode_rewards[0]
        self.assertTrue(len(rewards) == 100)
        evaluation_score = np.mean(rewards)
        self.assertTrue(.5 * env.get_max_reward() < evaluation_score <= env.get_max_reward())

    def test_sac_on_pendulum(self):
        """
        Creates an SAC-Agent and runs it on Pendulum.
        """
        env = OpenAIGymEnv("Pendulum-v0")
        agent = SACAgent.from_spec(
            config_from_path("configs/sac_agent_for_pendulum.json"),
            state_space=env.state_space,
            action_space=env.action_space
        )

        worker = SingleThreadedWorker(
            env_spec=lambda: env,
            agent=agent,
            worker_executes_preprocessing=False,
            render=self.is_windows
        )
        # Note: SAC is more computationally expensive.
        episodes = 50
        results = worker.execute_episodes(episodes)

        print(results)

        self.assertTrue(results["timesteps_executed"] == episodes * 200)
        self.assertTrue(results["episodes_executed"] == episodes)
        self.assertGreater(results["mean_episode_reward"], -800)

    def test_sac_on_cartpole(self):
        """
        Creates an SAC-Agent and runs it on CartPole.
        """
        env = OpenAIGymEnv("CartPole-v0")
        agent = SACAgent.from_spec(
            config_from_path("configs/sac_agent_for_cartpole.json"),
            state_space=env.state_space,
            action_space=env.action_space
        )

        worker = SingleThreadedWorker(
            env_spec=lambda: env,
            agent=agent,
            worker_executes_preprocessing=False,
            render=self.is_windows
        )

        time_steps = 10000
        results = worker.execute_timesteps(time_steps)

        print(results)

    def test_sac_2x2_grid_world_with_container_actions(self):
        """
        Creates a SAC agent and runs it via a Runner on a simple 2x2 GridWorld using container actions.
        """
        # ftj = forward + turn + jump
        env_spec = dict(world="2x2", action_type="ftj", state_representation="xy+orientation")
        dummy_env = GridWorld.from_spec(env_spec)
        agent_config = config_from_path("configs/sac_agent_for_2x2_gridworld_with_container_actions.json")
        preprocessing_spec = agent_config.pop("preprocessing_spec")

        agent = SACAgent.from_spec(
            agent_config,
            state_space=FloatBox(shape=(4,)),
            action_space=dummy_env.action_space,
        )

        time_steps = 10000
        worker = SingleThreadedWorker(
            env_spec=lambda: GridWorld.from_spec(env_spec),
            agent=agent,
            preprocessing_spec=preprocessing_spec,
            worker_executes_preprocessing=False,
            render=False
        )
        results = worker.execute_timesteps(time_steps, use_exploration=True)
        print(results)

    def test_sac_cartpole_on_ray(self):
        """
        Tests sac on Ape-X.
        """
        # Import Ray here so other test cases do not need to import it if not installed.
        from rlgraph.execution.ray import ApexExecutor
        env_spec = dict(
            type="openai",
            gym_env="CartPole-v0"
        )
        agent_config = config_from_path("configs/sac_cartpole_on_apex.json")
        executor = ApexExecutor(
            environment_spec=env_spec,
            agent_config=agent_config,
        )
        # Define executor, test assembly.
        print("Successfully created executor.")

        # Executes actual workload.
        result = executor.execute_workload(workload=dict(num_timesteps=20000, report_interval=1000,
                                                         report_interval_min_seconds=1))
        print("Finished executing workload:")
        print(result)
