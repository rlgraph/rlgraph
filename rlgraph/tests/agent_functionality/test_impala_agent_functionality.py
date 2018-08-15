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

import logging
import unittest
from rlgraph.agents import Agent
from rlgraph.components.neural_networks.policy import Policy
from rlgraph.components.papers.impala.large_impala_network import LargeIMPALANetwork
from rlgraph.environments import RandomEnv
from rlgraph.execution.single_threaded_worker import SingleThreadedWorker
from rlgraph.spaces import *
from rlgraph.tests.component_test import ComponentTest
from rlgraph.tests.test_util import config_from_path
from rlgraph.utils import root_logger


class TestIMPALAAgentFunctionality(unittest.TestCase):
    """
    Tests the LargeIMPALANetwork functionality and IMPALAAgent assembly on the RandomEnv.
    For details on the IMPALA algorithm, see [1]:

    [1] IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures - Espeholt, Soyer,
        Munos et al. - 2018 (https://arxiv.org/abs/1802.01561)
    """
    root_logger.setLevel(level=logging.INFO)

    # Use the exact same Spaces as in the IMPALA paper.
    action_space = IntBox(9, add_batch_rank=True, add_time_rank=True, time_major=True)
    input_space = Dict(
        image=FloatBox(shape=(96, 72, 3)), text=TextBox(),
        previous_action=FloatBox(shape=(9,)),
        previous_reward=FloatBox(shape=(1,)),  # add the extra rank for proper concatenating with the other inputs.
        add_batch_rank=True,
        add_time_rank=True,
        time_major=True
    )
    internal_states_space = Tuple(FloatBox(shape=(256,)), FloatBox(shape=(256,)), add_batch_rank=True)

    def test_large_impala_network_without_agent(self):
        """
        Creates a large IMPALA architecture network and runs an input sample through it.
        """
        # Create the network (with a small time-step value for this test).
        large_impala_architecture = LargeIMPALANetwork()
        test = ComponentTest(
            large_impala_architecture, input_spaces=dict(input_dict=self.input_space)
        )

        # Send a 2x3 sample through the network (2=sequence-length (time-rank), 3=batch-size).
        sample_input = self.input_space.sample(size=(2, 3))
        expected = None
        ret = test.test(("apply", sample_input), expected_outputs=expected)
        # Check shape of outputs.
        self.assertEquals(ret[0].shape, (2, 3, 256))
        # Check shapes of current internal_states (c and h).
        self.assertEquals(ret[1][0].shape, (3, 256))
        self.assertEquals(ret[1][1].shape, (3, 256))

    def test_large_impala_policy_without_agent(self):
        """
        Creates a large IMPALA architecture network inside a policy and runs an input sample through it.
        """
        # Create the network (with a small time-step value for this test).
        large_impala_architecture = LargeIMPALANetwork()
        # IMPALA uses a baseline action adapter (v-trace off-policy PG with baseline value function).
        policy = Policy(large_impala_architecture, action_space=self.action_space,
                        action_adapter_spec=dict(type="baseline_action_adapter"))
        test = ComponentTest(
            policy, input_spaces=dict(nn_input=self.input_space, internal_states=self.internal_states_space),
            action_space=self.action_space
        )

        # Send a 1x1 sample through the network (1=sequence-length (time-rank), 1=batch-size).
        nn_input = self.input_space.sample(size=(1, 1))
        initial_internal_states = self.internal_states_space.zeros(size=1)
        expected = None
        actions, last_internal_states = test.test(
            ("get_action", [nn_input, initial_internal_states]), expected_outputs=expected
        )
        print("First action: {}".format(actions))

        # Send another 1x1 sample through the network using the previous internal-state.
        next_nn_input = self.input_space.sample(size=(1, 1))
        expected = None
        actions, last_internal_states = test.test(("get_action", [next_nn_input, last_internal_states]),
                                                  expected_outputs=expected)
        print("Second action: {}".format(actions))

        # Send time x batch states through the network to simulate agent-type=learner behavior.
        next_nn_input = self.input_space.sample(size=(6, 1))  # time-steps=6, batch=1 (must match last-internal-states)
        expected = None
        actions, last_internal_states = test.test(("get_action", [next_nn_input, last_internal_states]),
                                                  expected_outputs=expected)
        print("Actions 3 to 8: {}".format(actions))


    # TODO move this to test_all_compile once it works.
    def test_impala_assembly(self):
        """
        Creates an IMPALAAgent and runs it for a few steps in the RandomEnv.
        """
        env = RandomEnv(state_space=FloatBox(shape=(2,)), action_space=IntBox(2), deterministic=True)
        agent = Agent.from_spec(
            config_from_path("configs/impala_agent_for_random_env.json"),
            state_space=env.state_space,
            action_space=env.action_space
        )

        worker = SingleThreadedWorker(environment=env, agent=agent)
        timesteps = 100
        results = worker.execute_timesteps(timesteps, use_exploration=False)

        print(results)

        self.assertEqual(results["timesteps_executed"], timesteps)
        self.assertEqual(results["env_frames"], timesteps)
        # Assert deterministic execution of Env and Agent.
        #self.assertAlmostEqual(results["mean_episode_reward"], 5.923551400230593)
        #self.assertAlmostEqual(results["max_episode_reward"], 14.312868008192979)
        #self.assertAlmostEqual(results["final_episode_reward"], 0.14325251090518198)
