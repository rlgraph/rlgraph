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

from rlgraph.execution.ray import ApexExecutor


class TestApexExecutor(unittest.TestCase):
    """
    Tests the ApexExecutor which provides an interface for distributing Apex-style workloads
    via Ray.
    """
    env_spec = dict(
      type="openai",
      gym_env="CartPole-v0"
    )

    def test_learning_cartpole(self):
        """
        Tests if apex can learn a simple environment using a single worker, thus replicating
        dqn.
        """
        path = os.path.join(os.getcwd(), "configs/apex_agent_cartpole.json")
        with open(path, 'rt') as fp:
            agent_config = json.load(fp)

        # Cartpole settings from cartpole dqn test.
        agent_config.update(
            update_spec=dict(update_interval=4, batch_size=24, sync_interval=64),
            optimizer_spec=dict(learning_rate=0.0002, clip_grad_norm=40.0)
        )

        executor = ApexExecutor(
            environment_spec=self.env_spec,
            agent_config=agent_config,
        )
        # Define executor, test assembly.
        print("Successfully created executor.")

        # Executes actual workload.
        result = executor.execute_workload(workload=dict(num_timesteps=100000, report_interval=1000))
        print("Finished executing workload:")
        print(result)

    # TODO this is long running, move to long running tess?


