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

import json
import os
import unittest
from time import sleep

from rlgraph import get_distributed_backend, spaces
from rlgraph.agents import Agent
from rlgraph.environments import RandomEnv
from rlgraph.execution.ray import RayWorker

if get_distributed_backend() == "ray":
    import ray


class TestRayWorker(unittest.TestCase):

    env_spec = dict(
      type="openai",
      gym_env="CartPole-v0"
    )
    agent_config = dict(
        type="random"
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
        worker = RayWorker.remote(self.env_spec, self.agent_config)

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
            if elem is True:
                terminals += 1
        self.assertEqual(terminals, 1)

        # Now run exactly 100 steps.
        task = worker.execute_and_get_timesteps.remote(100, break_on_terminal=False)
        sleep(5)
        # Retrieve result.
        result = ray.get(task)
        observations = result.get_batch()
        print('Task results, break on terminal = False:')
        print(observations)
        print(result.get_metrics())

        # We do not break on terminal so there should be exactly 100 steps.
        self.assertEqual(len(observations['terminals']), 100)

    def test_worker_weight_syncing(self):
        """
        Tests weight synchronization with a local agent and a remote worker.
        """
        # First, create a local agent
        env = RandomEnv(state_space=spaces.IntBox(2), action_space=spaces.IntBox(2), deterministic=True)
        path = os.path.join(os.getcwd(), "configs/apex_agent_cartpole.json")
        with open(path, 'rt') as fp:
            agent_config = json.load(fp)

        # Remove unneeded apex params.
        if "apex_replay_spec" in agent_config:
            agent_config.pop("apex_replay_spec")

        print(agent_config)
        local_agent = Agent.from_spec(
            agent_config,
            state_space=env.state_space,
            action_space=env.action_space
        )

        # Create a remote worker with the same agent config.
        worker = RayWorker.remote(self.env_spec, self.agent_config)

        # This imitates the initial executor sync without ray.put
        weights = local_agent.get_policy_weights()
        print('Weight type in init sync = {}'.format(type(weights)))
        print(weights)
        worker.set_policy_weights.remote(weights)
        print('Init weight sync successful.')

        # Replicate worker syncing steps as done in e.g. Ape-X executor:
        weights = ray.put(local_agent.get_policy_weights())
        print('Weight type returned by ray put = {}'.format(type(weights)))
        print(weights)
        worker.set_policy_weights.remote(weights)
        print('Object store weight sync successful.')
