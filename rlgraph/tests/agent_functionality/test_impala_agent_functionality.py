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
import numpy as np
import time
import unittest

from rlgraph.components.common.environment_stepper import EnvironmentStepper
from rlgraph.components.neural_networks.policy import Policy
from rlgraph.components.neural_networks.actor_component import ActorComponent
from rlgraph.components.papers.impala.large_impala_network import LargeIMPALANetwork
from rlgraph.components.explorations import Exploration
from rlgraph.environments import Environment, DeepmindLabEnv
from rlgraph.spaces import *
from rlgraph.utils.ops import DataOpTuple
from rlgraph.tests.component_test import ComponentTest
from rlgraph.tests.test_util import recursive_assert_almost_equal
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
    action_probs_space = FloatBox(shape=(9,), add_batch_rank=True, add_time_rank=True, time_major=True)
    input_space = Dict(
        RGB_INTERLEAVED=FloatBox(shape=(96, 72, 3)),
        INSTR=TextBox(),
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
        Creates a large IMPALA architecture network inside a policy and runs a few input samples through it.
        """
        # Create the network.
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
        self.assertEquals(actions.shape, (1, 1))
        self.assertEquals(last_internal_states[0].shape, (1, 256))
        self.assertEquals(last_internal_states[1].shape, (1, 256))

        # Send another 1x1 sample through the network using the previous internal-state.
        next_nn_input = self.input_space.sample(size=(1, 1))
        expected = None
        actions, last_internal_states = test.test(("get_action", [next_nn_input, last_internal_states]),
                                                  expected_outputs=expected)
        print("Second action: {}".format(actions))
        self.assertEquals(actions.shape, (1, 1))
        self.assertEquals(last_internal_states[0].shape, (1, 256))
        self.assertEquals(last_internal_states[1].shape, (1, 256))

        # Send time x batch states through the network to simulate agent-type=learner behavior.
        next_nn_input = self.input_space.sample(size=(6, 1))  # time-steps=6, batch=1 (must match last-internal-states)
        expected = None
        actions, last_internal_states = test.test(("get_action", [next_nn_input, last_internal_states]),
                                                  expected_outputs=expected)
        print("Actions 3 to 8: {}".format(actions))
        self.assertEquals(actions.shape, (6, 1))
        self.assertEquals(last_internal_states[0].shape, (1, 256))
        self.assertEquals(last_internal_states[1].shape, (1, 256))

    def test_large_impala_actor_component_without_agent(self):
        """
        Creates a large IMPALA architecture network inside a policy inside an actor component and runs a few input
        samples through it.
        """
        batch_size = 4

        # Use IMPALA paper's preprocessor of division by 255 (only for the Image).
        preprocessor_spec_for_actor_component = dict(RGB_INTERLEAVED=[dict(type="divide", divisor=255)])
        # IMPALA uses a baseline action adapter (v-trace off-policy PG with baseline value function).
        policy = Policy(LargeIMPALANetwork(), action_space=self.action_space,
                        action_adapter_spec=dict(type="baseline_action_adapter"))
        exploration = Exploration(epsilon_spec=dict(decay_spec=dict(
            type="linear_decay", from_=1.0, to_=0.1, start_timestep=0, num_timesteps=100)
        ))
        actor_component = ActorComponent(preprocessor_spec_for_actor_component, policy, exploration)

        test = ComponentTest(
            actor_component, input_spaces=dict(
                states=self.input_space,
                internal_states=self.internal_states_space
            ),
            action_space=self.action_space
        )

        # Send a sample through the network (sequence-length (time-rank) x batch-size).
        nn_dict_input = self.input_space.sample(size=(1, batch_size))
        initial_internal_states = self.internal_states_space.zeros(size=batch_size)
        expected = None
        preprocessed_states, actions, last_internal_states = test.test(
            ("get_preprocessed_state_and_action", [nn_dict_input, initial_internal_states]), expected_outputs=expected
        )
        print("First action: {}".format(actions))
        self.assertEquals(actions.shape, (1, batch_size,))
        self.assertEquals(last_internal_states[0].shape, (batch_size, 256))
        self.assertEquals(last_internal_states[1].shape, (batch_size, 256))
        # Check preprocessed state (all the same except 'image' channel).
        recursive_assert_almost_equal(
            preprocessed_states, dict(
                RGB_INTERLEAVED=nn_dict_input["RGB_INTERLEAVED"] / 255,
                INSTR=nn_dict_input["INSTR"],
                previous_action=nn_dict_input["previous_action"],
                previous_reward=nn_dict_input["previous_reward"],
            )
        )

        # Send another 1x1 sample through the network using the previous internal-state.
        next_nn_input = self.input_space.sample(size=(1, batch_size))
        expected = None
        preprocessed_states, actions, last_internal_states = test.test(
            ("get_preprocessed_state_and_action", [next_nn_input, last_internal_states]), expected_outputs=expected
        )
        print("Second action: {}".format(actions))
        self.assertEquals(actions.shape, (1, batch_size))
        self.assertEquals(last_internal_states[0].shape, (batch_size, 256))
        self.assertEquals(last_internal_states[1].shape, (batch_size, 256))
        # Check preprocessed state (all the same except 'image' channel, which gets divided by 255).
        recursive_assert_almost_equal(
            preprocessed_states, dict(
                RGB_INTERLEAVED=next_nn_input["RGB_INTERLEAVED"] / 255,
                INSTR=next_nn_input["INSTR"],
                previous_action=next_nn_input["previous_action"],
                previous_reward=next_nn_input["previous_reward"],
            )
        )

        # Send time x batch states through the network to simulate agent-type=learner behavior.
        # time-steps=20, batch=1 (must match last-internal-states)
        next_nn_input = self.input_space.sample(size=(20, batch_size))
        expected = None
        preprocessed_states, actions, last_internal_states = test.test(
            ("get_preprocessed_state_and_action", [next_nn_input, last_internal_states]), expected_outputs=expected
        )
        print("Actions 3 to 22: {}".format(actions))
        self.assertEquals(actions.shape, (20, batch_size))
        self.assertEquals(last_internal_states[0].shape, (batch_size, 256))
        self.assertEquals(last_internal_states[1].shape, (batch_size, 256))
        # Check preprocessed state (all the same except 'image' channel).
        recursive_assert_almost_equal(
            preprocessed_states, dict(
                RGB_INTERLEAVED=next_nn_input["RGB_INTERLEAVED"] / 255,
                INSTR=next_nn_input["INSTR"],
                previous_action=next_nn_input["previous_action"],
                previous_reward=next_nn_input["previous_reward"],
            )
        )

    def test_environment_stepper_component_with_large_impala_architecture(self):
        env_spec = dict(
            type="deepmind_lab", level_id="seekavoid_arena_01", observations=["RGB_INTERLEAVED", "INSTR"],
            frameskip=4
        )
        dummy_env = Environment.from_spec(env_spec)
        state_space = dummy_env.state_space
        action_space = dummy_env.action_space
        actor_component = ActorComponent(
            # Preprocessor spec (only for image and prev-action channel).
            dict(
                # The images from the env  are divided by 255.
                RGB_INTERLEAVED=[dict(type="divide", divisor=255)],
                # The prev. actions from the env must be flattened.
                previous_action=[dict(type="reshape", flatten=True, flatten_categories=action_space.num_categories)]
            ),
            # Policy spec.
            dict(neural_network=LargeIMPALANetwork(), action_space=action_space),
            # Exploration spec.
            Exploration(epsilon_spec=dict(decay_spec=dict(
                type="linear_decay", from_=1.0, to_=0.1, start_timestep=0, num_timesteps=100)
            ))
        )
        environment_stepper = EnvironmentStepper(
            environment_spec=env_spec,
            actor_component_spec=actor_component,
            state_space=state_space,
            reward_space="float64",
            # Add both prev-action and -reward into the state sent through the network.
            add_previous_action=True,
            add_previous_reward=True,
            add_action_probs=True,
            action_probs_space=self.action_probs_space
        )

        test = ComponentTest(
            component=environment_stepper,
            input_spaces=dict(
                internal_states=self.internal_states_space,
                num_steps=int,
                time_step=int
            ),
            action_space=action_space,
            disable_monitoring=True,  # Make session-creation hang in docker.
        )

        # Start Specifiable Server with Env manually.
        environment_stepper.environment_server.start()

        # Reset the stepper.
        test.test("reset")

        initial_internal_states = self.internal_states_space.zeros(size=1)

        # Step n times through the Env and collect results.
        # 1st return value is the step-op (None), 2nd return value is the tuple of items (3 steps each), with each
        # step containing: Preprocessed state, actions, rewards, episode returns, terminals, (raw) next-states.
        time_steps = 500
        time_start = time.monotonic()
        out = test.test(("step", [initial_internal_states, time_steps, 0]), expected_outputs=None)
        time_end = time.monotonic()
        print("Done running {} steps in Deepmind Lab env using IMPALA network in {}sec.".format(
            time_steps, time_end - time_start)
        )

        # Check types of outputs.
        self.assertTrue(out[0] is None)  # the step op (no_op).
        self.assertTrue(isinstance(out[1], DataOpTuple))  # the step results as a tuple (see below)

        # Check types of single data.
        self.assertTrue(out[1][0]["INSTR"].dtype == np.object)
        self.assertTrue(out[1][0]["RGB_INTERLEAVED"].dtype == np.float32)
        self.assertTrue(out[1][0]["RGB_INTERLEAVED"].min() >= 0.0)  # make sure we have pixels / 255
        self.assertTrue(out[1][0]["RGB_INTERLEAVED"].max() <= 1.0)
        self.assertTrue(out[1][1].dtype == np.int32)  # actions
        self.assertTrue(out[1][2].dtype == np.float64)  # rewards
        self.assertTrue(out[1][3].dtype == np.float64)  # episode return
        self.assertTrue(out[1][4].dtype == np.bool_)  # next-state is terminal?
        self.assertTrue(out[1][5]["INSTR"].dtype == np.object)  # next state (raw, not preprocessed)
        self.assertTrue(out[1][5]["RGB_INTERLEAVED"].dtype == np.uint8)  # next state (raw, not preprocessed)
        self.assertTrue(out[1][5]["RGB_INTERLEAVED"].min() >= 0)  # make sure we have pixels
        self.assertTrue(out[1][5]["RGB_INTERLEAVED"].max() <= 255)
        # action probs (test whether sum to one).
        self.assertTrue(out[1][6].dtype == np.float32)
        self.assertTrue(out[1][6].min() >= 0.0)
        self.assertTrue(out[1][6].max() <= 1.0)
        recursive_assert_almost_equal(out[1][6].sum(axis=-1, keepdims=False), np.ones(shape=(time_steps,)), decimals=4)
        # internal states (c- and h-state)
        self.assertTrue(out[1][7][0].dtype == np.float32)
        self.assertTrue(out[1][7][1].dtype == np.float32)
        self.assertTrue(out[1][7][0].shape == (time_steps, 1, 256))
        self.assertTrue(out[1][7][1].shape == (time_steps, 1, 256))

        # Check whether episode returns match single rewards (including terminal signals).
        episode_returns = 0.0
        for i in range(time_steps):
            episode_returns += out[1][2][i]
            self.assertAlmostEqual(episode_returns, out[1][3][i])
            # Terminal: Reset for next step.
            if out[1][4][i] is np.bool_(True):
                episode_returns = 0.0

        # Make sure we close the specifiable server (as we have started it manually and have no monitored session).
        environment_stepper.environment_server.stop()
        test.terminate()

