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

from rlgraph.execution.ray.ray_policy_worker import RayPolicyWorker
from rlgraph.execution.ray.ray_util import RayWeight
from rlgraph.tests.test_util import config_from_path

from rlgraph import get_distributed_backend
from rlgraph.agents import Agent
from rlgraph.environments import Environment

if get_distributed_backend() == "ray":
    import ray


class TestRayPolicyWorker(unittest.TestCase):

    env_spec = dict(
      type="openai",
      gym_env="CartPole-v0"
    )

    def setUp(self):
        """
        Inits a local redis and scheduler.
        """
        ray.init()

    def test_policy_and_vf_weight_syncing(self):
        """
        Tests weight synchronization with a local agent and a remote worker.
        """
        # First, create a local agent
        env_spec = dict(
            type="openai",
            gym_env="CartPole-v0"
        )
        env = Environment.from_spec(env_spec)
        agent_config = config_from_path("configs/sync_batch_ppo_cartpole.json")

        ray_spec = agent_config["execution_spec"].pop("ray_spec")
        local_agent = Agent.from_spec(
            agent_config,
            state_space=env.state_space,
            action_space=env.action_space
        )
        ray_spec["worker_spec"]["worker_sample_size"] = 50
        # Create a remote worker with the same agent config.
        worker = RayPolicyWorker.as_remote().remote(agent_config, ray_spec["worker_spec"],
                                                   self.env_spec)

        # This imitates the initial executor sync without ray.put
        weights = RayWeight(local_agent.get_weights())
        print('Weight type in init sync = {}'.format(type(weights)))
        print("Weights = ", weights)
        worker.set_weights.remote(weights)
        print('Init weight sync successful.')

        # Replicate worker syncing steps as done in e.g. Ape-X executor:
        weights = RayWeight(local_agent.get_weights())
        print('Weight type returned by ray put = {}'.format(type(weights)))
        print(weights)
        ret = worker.set_weights.remote(weights)
        ray.wait([ret])
        print('Object store weight sync successful.')

