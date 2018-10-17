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
from copy import deepcopy

import numpy as np

from rlgraph import get_backend
from rlgraph.components import PreprocessorStack
from rlgraph.environments import OpenAIGymEnv
from rlgraph.execution.ray.apex import ApexExecutor
from rlgraph.tests.test_util import config_from_path, recursive_assert_almost_equal


class TestApexExecutor(unittest.TestCase):
    """
    Tests the ApexExecutor which provides an interface for distributing Apex-style workloads
    via Ray.
    """
    def test_learning_2x2_grid_world(self):
        """
        Tests if apex can learn a simple environment using a single worker, thus replicating
        dqn.
        """
        env_spec = dict(
            type="grid-world",
            world="2x2",
            save_mode=False
        )
        agent_config = config_from_path("configs/apex_agent_gridworld_for_2x2_grid.json")
        # TODO remove after unified backends
        if get_backend() == "pytorch":
            agent_config["memory_spec"]["type"] = "mem_prioritized_replay"
        executor = ApexExecutor(
            environment_spec=env_spec,
            agent_config=agent_config,
        )
        # Define executor, test assembly.
        print("Successfully created executor.")

        # Executes actual workload.
        result = executor.execute_workload(workload=dict(
            num_timesteps=5000, report_interval=100, report_interval_min_seconds=1)
        )
        full_worker_stats = executor.result_by_worker()
        print("All finished episode rewards")
        print(full_worker_stats["episode_rewards"])

        print("STATES:\n{}".format(executor.local_agent.last_q_table["states"]))
        print("\n\nQ(s,a)-VALUES:\n{}".format(np.round_(executor.local_agent.last_q_table["q_values"], decimals=2)))

        # Check q-table for correct values.
        expected_q_values_per_state = {
            (1.0, 0, 0, 0): (-1, -5, 0, -1),
            (0, 1.0, 0, 0): (-1, 1, 0, 0)
        }
        for state, q_values in zip(
                executor.local_agent.last_q_table["states"], executor.local_agent.last_q_table["q_values"]
        ):
            state, q_values = tuple(state), tuple(q_values)
            assert state in expected_q_values_per_state, \
                "ERROR: state '{}' not expected in q-table as it's a terminal state!".format(state)
            recursive_assert_almost_equal(q_values, expected_q_values_per_state[state], decimals=0)

    def test_learning_cartpole(self):
        """
        Tests if apex can learn a simple environment using a single worker, thus replicating
        dqn.
        """
        env_spec = dict(
            type="openai",
            gym_env="CartPole-v0"
        )
        agent_config = config_from_path("configs/apex_agent_cartpole.json")
        # TODO remove after unified backends
        if get_backend() == "pytorch":
            agent_config["memory_spec"]["type"] = "mem_prioritized_replay"

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

    def test_learning_cartpole_n_step(self):
        """
        Tests if apex can learn a simple environment using a single worker, thus replicating
        DQN.
        """
        env_spec = dict(
            type="openai",
            gym_env="CartPole-v0"
        )
        agent_config = config_from_path("configs/apex_agent_cartpole.json")

        # Use n-step adjustments.
        agent_config["execution_spec"]["ray_spec"]["worker_spec"]["n_step_adjustment"] = 3
        agent_config["execution_spec"]["ray_spec"]["apex_replay_spec"]["n_step_adjustment"] = 3
        agent_config["n_step"] = 3

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

    def test_with_final_eval(self):
        """
        Tests if apex can learn a simple environment using a single worker, thus replicating
        DQN.
        """
        env_spec = dict(
            type="openai",
            gym_env="CartPole-v0"
        )
        agent_config = config_from_path("configs/apex_agent_cartpole.json")

        # Use n-step adjustments.
        agent_config["execution_spec"]["ray_spec"]["worker_spec"]["n_step_adjustment"] = 3
        agent_config["execution_spec"]["ray_spec"]["apex_replay_spec"]["n_step_adjustment"] = 3
        agent_config["n_step"] = 3

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

        # Get agent.
        agent = executor.local_agent
        preprocessing_spec = agent_config["preprocessing_spec"]

        # Create env.
        env = OpenAIGymEnv.from_spec(env_spec)

        if preprocessing_spec is not None:
            preprocessing_spec = deepcopy(preprocessing_spec)
            in_space = env.state_space.with_batch_rank()
            in_space = deepcopy(in_space)
            # Set scopes.
            scopes = [preprocessor["scope"] for preprocessor in preprocessing_spec]
            # Set backend to python.
            for spec in preprocessing_spec:
                spec["backend"] = "python"
            processor_stack = PreprocessorStack(*preprocessing_spec, backend="python")
            build_space = in_space
            for sub_comp_scope in scopes:
                processor_stack.sub_components[sub_comp_scope].create_variables(input_spaces=dict(
                    preprocessing_inputs=build_space
                ), action_space=None)
                build_space = processor_stack.sub_components[sub_comp_scope].get_preprocessed_space(build_space)
            processor_stack.reset()
        else:
            processor_stack = None

        ep_rewards = []
        print("finished learning, starting eval")
        for _ in range(10):
            state = env.reset()
            terminal = False
            ep_reward = 0
            while not terminal:
                state = agent.state_space.force_batch(state)
                if processor_stack is not None:
                    state = processor_stack.preprocess(state)

                actions = agent.get_action(states=state, use_exploration=False, apply_preprocessing=False)
                next_state, step_reward, terminal, info = env.step(actions=actions[0])
                ep_reward += step_reward

                state = next_state
                if terminal:
                    ep_rewards.append(ep_reward)
                    break

        print("Eval episode rewards:")
        print(ep_rewards)
