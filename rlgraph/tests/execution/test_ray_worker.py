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
from time import sleep
from rlgraph.tests.test_util import recursive_assert_almost_equal, config_from_path
import numpy as np

from rlgraph import get_distributed_backend
from rlgraph.agents import Agent
from rlgraph.environments import Environment
from rlgraph.execution.ray import RayWorker

if get_distributed_backend() == "ray":
    import ray


class TestRayWorker(unittest.TestCase):

    env_spec = dict(
      type="openai",
      gym_env="CartPole-v0"
    )

    def setUp(self):
        """
        Inits a local redis and scheduler.
        """
        ray.init()

    def test_get_timesteps(self):
        """
        Simply tests if time-step execution loop works and returns the samples.
        """
        agent_config = config_from_path("configs/apex_agent_cartpole.json")
        ray_spec = agent_config["execution_spec"].pop("ray_spec")
        ray_spec["worker_spec"]["worker_sample_size"] = 100
        worker = RayWorker.as_remote().remote(agent_config, ray_spec["worker_spec"], self.env_spec,  auto_build=True)

        # Test when breaking on terminal.
        # Init remote task.
        task = worker.execute_and_get_timesteps.remote(100, break_on_terminal=True)
        sleep(5)
        # Retrieve result.
        result = ray.get(task)
        observations = result.get_batch()
        print('Task results, break on terminal = True:')
        print(observations)
        print(result.get_metrics())

        self.assertLessEqual(len(observations['terminals']), 100)
        # There can only be one terminal in there because we break on terminals:
        terminals = 0
        for elem in observations['terminals']:
            if np.alltrue(elem):
                terminals += 1
        print(observations['terminals'])
        self.assertEqual(1, terminals)

        # Now run exactly 100 steps.
        task = worker.execute_and_get_timesteps.remote(100, break_on_terminal=False)
        sleep(5)
        # Retrieve result.
        result = ray.get(task)
        observations = result.get_batch()
        print('Task results, break on terminal = False:')
        print(result.get_metrics())

        # We do not break on terminal so there should be exactly 100 steps.
        self.assertEqual(len(observations['terminals']), 100)

        # Test with count.
        task = worker.execute_and_get_with_count.remote()
        # Retrieve result.
        result, size = ray.get(task)
        observations = result.get_batch()
        print("Returned exact count: ")
        print(size)

        # We do not break on terminal so there should be exactly 100 steps.
        self.assertEqual(len(observations["terminals"]), size)

    def test_metrics(self):
        """
        Tests metric collection for 1 and multiple environments.
        """
        agent_config = config_from_path("configs/apex_agent_cartpole.json")

        ray_spec = agent_config["execution_spec"].pop("ray_spec")
        ray_spec["worker_spec"]["worker_sample_size"] = 100
        worker_spec = ray_spec["worker_spec"]
        worker = RayWorker.as_remote().remote(agent_config, ray_spec["worker_spec"], self.env_spec,  auto_build=True)

        print("Testing statistics for 1 environment:")
        # Run for a while:
        task = worker.execute_and_get_timesteps.remote(100, break_on_terminal=False)
        sleep(1)
        # Include a transition between calls.
        task = worker.execute_and_get_timesteps.remote(100, break_on_terminal=False)
        sleep(1)
        # Retrieve result.
        result = ray.get(task)
        print('Task results:')
        print(result.get_metrics())

        # Get worker metrics.
        task = worker.get_workload_statistics.remote()
        result = ray.get(task)
        print("Worker statistics:")

        # In cartpole, num timesteps = reward -> must be the same.
        print("Cartpole episode rewards: {}".format(result["episode_rewards"]))
        print("Cartpole episode timesteps: {}".format(result["episode_timesteps"]))
        recursive_assert_almost_equal(result["episode_rewards"], result["episode_timesteps"])

        # Now repeat this but for multiple environments.
        print("Testing statistics for 4 environments:")
        worker_spec["num_worker_environments"] = 4
        worker_spec["num_background_environments"] = 2
        worker = RayWorker.as_remote().remote(agent_config, ray_spec["worker_spec"], self.env_spec,  auto_build=True)

        task = worker.execute_and_get_timesteps.remote(100, break_on_terminal=False)
        sleep(1)
        result = ray.get(task)
        task = worker.execute_and_get_timesteps.remote(100, break_on_terminal=False)
        sleep(1)
        result = ray.get(task)
        task = worker.get_workload_statistics.remote()
        result = ray.get(task)
        print("Multi-env statistics:")
        print("Cartpole episode rewards: {}".format(result["episode_rewards"]))
        print("Cartpole episode timesteps: {}".format(result["episode_timesteps"]))
        recursive_assert_almost_equal(result["episode_rewards"], result["episode_timesteps"])

    def test_worker_weight_syncing(self):
        """
        Tests weight synchronization with a local agent and a remote worker.
        """
        # First, create a local agent
        env_spec = dict(
            type="openai",
            gym_env="PongNoFrameskip-v4",
            # The frameskip in the agent config will trigger worker skips, this
            # is used for internal env.
            frameskip=4,
            max_num_noops=30,
            episodic_life=True
        )
        env = Environment.from_spec(env_spec)
        agent_config = config_from_path("configs/ray_apex_for_pong.json")

        # Remove unneeded apex params.
        if "apex_replay_spec" in agent_config:
            agent_config.pop("apex_replay_spec")

        ray_spec = agent_config["execution_spec"].pop("ray_spec")
        local_agent = Agent.from_spec(
            agent_config,
            state_space=env.state_space,
            action_space=env.action_space
        )

        # Create a remote worker with the same agent config.
        worker = RayWorker.as_remote().remote(agent_config, ray_spec["worker_spec"], self.env_spec,  auto_build=True)

        # This imitates the initial executor sync without ray.put
        weights = local_agent.get_policy_weights()
        print('Weight type in init sync = {}'.format(type(weights)))
        worker.set_policy_weights.remote(weights)
        print('Init weight sync successful.')

        # Replicate worker syncing steps as done in e.g. Ape-X executor:
        weights = ray.put(local_agent.get_policy_weights())
        print('Weight type returned by ray put = {}'.format(type(weights)))
        print(weights)
        worker.set_policy_weights.remote(weights)
        print('Object store weight sync successful.')

