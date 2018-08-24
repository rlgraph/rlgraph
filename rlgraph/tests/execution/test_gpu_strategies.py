# Copyright 2018 The RLgraph authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, softwamre
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
from rlgraph.utils import root_logger
from logging import DEBUG

from rlgraph.agents import ApexAgent
from rlgraph.environments import OpenAIGymEnv
from rlgraph.execution.ray import ApexExecutor
from rlgraph.tests.test_util import config_from_path


class TestGpuStrategies(unittest.TestCase):
    """
    Tests gpu strategies.
    """
    env_spec = dict(
        type="openai",
        gym_env="PongNoFrameskip-v4",
        # The frameskip in the agent config will trigger worker skips, this
        # is used for internal env.
        frameskip=4,
        max_num_noops=30,
        episodic_life=True
    )

    def test_multi_gpu_agent_compilation(self):
        """
        Tests if the multi gpu strategy can compile successfully on a multi gpu system.

        THIS TEST REQUIRES A MULTI GPU SYSTEM.
        """
        root_logger.setLevel(DEBUG)
        agent_config = config_from_path("configs/multi_gpu_ray_apex_for_pong.json")
        agent_config["execution_spec"].pop("ray_spec")
        environment = OpenAIGymEnv("Pong-v0", frameskip=4)

        agent = ApexAgent.from_spec(
            agent_config, state_space=environment.state_space, action_space=environment.action_space
        )
        print('Compiled apex agent')

    def test_apex_multi_gpu_update(self):
        """
        Tests if the multi GPU optimizer can perform successful updates, using the apex executor.
        """
        agent_config = config_from_path("configs/multi_gpu_ray_apex_for_pong.json")
        executor = ApexExecutor(
            environment_spec=self.env_spec,
            agent_config=agent_config,
        )

        # Executes actual workload.
        result = executor.execute_workload(workload=dict(
            num_timesteps=100000, report_interval=10000, report_interval_min_seconds=10)
        )


    # TODO (Bart maybe): We should probably have some tests that simply test the update call
    # This is just slightly annoying because we have to assemble a preprocessed batch manually
    # It would be good to have a utility method for that to use in tests (e.g. sample atari batches).