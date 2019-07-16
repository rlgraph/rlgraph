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

from __future__ import absolute_import, division, print_function

import logging
import os
import unittest

import numpy as np
from scipy import stats

from rlgraph.agents.sac_agent import SACAlgorithmComponent, SACAgent
from rlgraph.components import Policy, QFunction, ReplayMemory, AdamOptimizer, \
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
        q_function = QFunction.from_spec(config["q_function"])

        agent_component = SACAlgorithmComponent(
            agent=None,
            policy_spec=policy,
            q_function_spec=q_function,
            preprocessing_spec=None,
            memory_spec=ReplayMemory.from_spec(config["memory"]),
            discount=config["discount"],
            initial_alpha=config["initial_alpha"],
            target_entropy=None,
            optimizer_spec=AdamOptimizer.from_spec(config["optimizer_spec"]),
            q_function_optimizer_spec=AdamOptimizer.from_spec(config["q_function_optimizer_spec"]),
            num_q_functions=2
        )

        test = ComponentTest(
            component=agent_component,
            input_spaces=dict(
                states=state_space.with_batch_rank(),
                preprocessed_states=state_space.with_batch_rank(),
                env_actions=continuous_action_space.with_batch_rank(),
                rewards=FloatBox(add_batch_rank=True),
                next_states=state_space.with_batch_rank(),
                terminals=terminal_space,
                importance_weights=FloatBox(add_batch_rank=True),
                deterministic=bool,
                policy_weights="variables:{}".format(policy.scope),
                time_percentage=float,
                # TODO: how to provide the space for multiple component variables?
                # q_weights=Dict(
                #    q_0="variables:{}".format(q_function.scope),
                #    q_1="variables:{}".format(agent_component._q_functions[1].scope),
                # )
            ),
            action_space=continuous_action_space,
            build_kwargs=dict(
                optimizer=agent_component.optimizer,
                build_options=dict(
                    vf_optimizer=agent_component.q_function_optimizer,
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

        action_sample = test.test(
            ("get_actions_from_preprocessed_states", [state_space.sample(batch_size), False])
        )["actions"]
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

        worker = SingleThreadedWorker(env_spec=lambda: env, agent=agent, update_rules=dict(update_every_n_units=1))
        worker.execute_episodes(num_episodes=500)
        rewards = worker.finished_episode_returns[0]  # 0=1st env in vector-env
        self.assertTrue(np.mean(rewards[:100]) < np.mean(rewards[-100:]))

        worker.execute_episodes(num_episodes=100, use_exploration=False)
        rewards = worker.finished_episode_returns[0]
        self.assertTrue(len(rewards) == 100)
        evaluation_score = np.mean(rewards)
        self.assertTrue(.5 * env.get_max_reward() < evaluation_score <= env.get_max_reward())

    def test_sac_on_pendulum(self):
        """
        Creates an SAC-Agent and runs it on Pendulum.
        """
        env_spec = dict(type="openai-gym", gym_env="Pendulum-v0")
        dummy_env = OpenAIGymEnv.from_spec(env_spec)
        agent = SACAgent.from_spec(
            config_from_path("configs/sac_agent_for_pendulum.json"),
            state_space=dummy_env.state_space,
            action_space=dummy_env.action_space
        )

        worker = SingleThreadedWorker(
            env_spec=env_spec,
            agent=agent,
            worker_executes_preprocessing=False,
            render=False,  # self.is_windows
            episode_finish_callback=lambda episode_return, duration, timesteps, **kwargs:
            print("episode: return={} ts={}".format(episode_return, timesteps)),
            update_rules=dict(unit="time_steps", update_every_n_units=1)
        )
        # Note: SAC is more computationally expensive.
        episodes = 100
        results = worker.execute_episodes(episodes)

        print(results)

        self.assertTrue(results["timesteps_executed"] == episodes * 200)
        self.assertTrue(results["episodes_executed"] == episodes)
        self.assertGreater(results["mean_episode_reward_last_10_episodes"], -400)
        self.assertGreater(results["max_episode_reward"], -200)

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
            render=False,  # self.is_windows,
            episode_finish_callback=lambda episode_return, duration, timesteps, **kwargs:
            print("episode: return={} ts={}".format(episode_return, timesteps))
        )

        time_steps = 5000
        results = worker.execute_timesteps(time_steps)

        print(results)

        self.assertTrue(results["timesteps_executed"] == time_steps)
        self.assertLessEqual(results["episodes_executed"], time_steps / 20)
        self.assertGreater(results["mean_episode_reward"], 40.0)
        self.assertGreater(results["max_episode_reward"], 100.0)
        self.assertGreater(results["mean_episode_reward_last_10_episodes"], 100.0)

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
