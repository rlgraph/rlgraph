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
from rlgraph.components.papers.impala.impala_networks import LargeIMPALANetwork
from rlgraph.components.explorations import Exploration
from rlgraph.environments import Environment
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
    action_probs_space = FloatBox(shape=(9,), add_batch_rank=True)
    input_space = Dict(
        RGB_INTERLEAVED=FloatBox(shape=(96, 72, 3)),
        INSTR=TextBox(),
        previous_action=FloatBox(shape=(9,)),
        previous_reward=FloatBox(shape=(1,)),  # add the extra rank for proper concatenating with the other inputs.
        add_batch_rank=True,
        add_time_rank=True,
        time_major=False
    )
    internal_states_space = Tuple(FloatBox(shape=(256,)), FloatBox(shape=(256,)), add_batch_rank=True)
    cluster_spec = dict(learner=["localhost:22222"], actor=["localhost:22223"])

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
        next_nn_input = self.input_space.sample(size=(1, 6))  # batch=1, time-steps=6 (must match last-internal-states)
        expected = None
        actions, last_internal_states = test.test(("get_action", [next_nn_input, last_internal_states]),
                                                  expected_outputs=expected)
        print("Actions 3 to 8: {}".format(actions))
        self.assertEquals(actions.shape, (1, 6))
        self.assertEquals(last_internal_states[0].shape, (1, 256))
        self.assertEquals(last_internal_states[1].shape, (1, 256))

    def test_large_impala_actor_component_without_agent(self):
        """
        Creates a large IMPALA architecture network inside a policy inside an actor component and runs a few input
        samples through it.
        """
        batch_size = 4

        # Use IMPALA paper's preprocessor of division by 255 (only for the Image).
        preprocessor_spec_for_actor_component = dict(
            type="dict-preprocessor-stack",
            preprocessors=dict(
                RGB_INTERLEAVED=[dict(type="divide", divisor=255)]
            )
        )
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
        nn_dict_input = self.input_space.sample(size=(batch_size, 1))
        initial_internal_states = self.internal_states_space.zeros(size=batch_size)
        expected = None
        preprocessed_states, actions, last_internal_states = test.test(
            ("get_preprocessed_state_and_action", [nn_dict_input, initial_internal_states]), expected_outputs=expected
        )
        print("First action: {}".format(actions))
        self.assertEquals(actions.shape, (batch_size, 1))
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
        next_nn_input = self.input_space.sample(size=(batch_size, 1))
        expected = None
        preprocessed_states, actions, last_internal_states = test.test(
            ("get_preprocessed_state_and_action", [next_nn_input, last_internal_states]), expected_outputs=expected
        )
        print("Second action: {}".format(actions))
        self.assertEquals(actions.shape, (batch_size, 1))
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
        next_nn_input = self.input_space.sample(size=(batch_size, 20))
        expected = None
        preprocessed_states, actions, last_internal_states = test.test(
            ("get_preprocessed_state_and_action", [next_nn_input, last_internal_states]), expected_outputs=expected
        )
        print("Actions 3 to 22: {}".format(actions))
        self.assertEquals(actions.shape, (batch_size, 20))
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
                type="dict-preprocessor-stack",
                preprocessors=dict(
                    ## The images from the env  are divided by 255.
                    #RGB_INTERLEAVED=[dict(type="divide", divisor=255)],
                    # The prev. action/reward from the env must be flattened/bumped-up-to-(1,).
                    previous_action=[dict(type="reshape", flatten=True, flatten_categories=action_space.num_categories)],
                    previous_reward=[dict(type="reshape", new_shape=(1,)), dict(type="convert_type", to_dtype="float32")],
                )
            ),
            # Policy spec.
            dict(network_spec=LargeIMPALANetwork(), action_space=action_space),
            # Exploration spec.
            Exploration(epsilon_spec=dict(decay_spec=dict(
                type="linear_decay", from_=1.0, to_=0.1, start_timestep=0, num_timesteps=100)
            ))
        )
        environment_stepper = EnvironmentStepper(
            environment_spec=env_spec,
            actor_component_spec=actor_component,
            state_space=state_space,
            reward_space="float32",
            internal_states_space=self.internal_states_space,
            num_steps=100,
            # Add both prev-action and -reward into the state sent through the network.
            add_previous_action=True,
            add_previous_reward=True,
            add_action_probs=True,
            action_probs_space=self.action_probs_space
        )

        test = ComponentTest(
            component=environment_stepper,
            action_space=action_space,
        )
        # Reset the stepper.
        test.test("reset")

        # Step n times through the Env and collect results.
        # 1st return value is the step-op (None), 2nd return value is the tuple of items (3 steps each), with each
        # step containing: Preprocessed state, actions, rewards, episode returns, terminals, (raw) next-states.
        time_start = time.perf_counter()
        steps = 10
        for _ in range(steps):
            out = test.test("step")
        time_total = time.perf_counter() - time_start
        print("Done running {}x{} steps in Deepmind Lab env using IMPALA network in {}sec ({} actions/sec).".format(
            steps, environment_stepper.num_steps, time_total , environment_stepper.num_steps * steps / time_total)
        )

        # Check types of outputs.
        self.assertTrue(isinstance(out, DataOpTuple))  # the step results as a tuple (see below)

        # Check types of single data.
        self.assertTrue(out[0]["INSTR"].dtype == np.object)
        self.assertTrue(out[0]["RGB_INTERLEAVED"].dtype == np.float32)
        self.assertTrue(out[0]["RGB_INTERLEAVED"].min() >= 0.0)  # make sure we have pixels / 255
        self.assertTrue(out[0]["RGB_INTERLEAVED"].max() <= 1.0)
        self.assertTrue(out[1].dtype == np.int32)  # actions
        self.assertTrue(out[2].dtype == np.float32)  # rewards
        self.assertTrue(out[3].dtype == np.float32)  # episode return
        self.assertTrue(out[4].dtype == np.bool_)  # next-state is terminal?
        self.assertTrue(out[5]["INSTR"].dtype == np.object)  # next state (raw, not preprocessed)
        self.assertTrue(out[5]["RGB_INTERLEAVED"].dtype == np.uint8)  # next state (raw, not preprocessed)
        self.assertTrue(out[5]["RGB_INTERLEAVED"].min() >= 0)  # make sure we have pixels
        self.assertTrue(out[5]["RGB_INTERLEAVED"].max() <= 255)
        # action probs (test whether sum to one).
        self.assertTrue(out[6].dtype == np.float32)
        self.assertTrue(out[6].min() >= 0.0)
        self.assertTrue(out[6].max() <= 1.0)
        recursive_assert_almost_equal(out[6].sum(axis=-1, keepdims=False),
                                      np.ones(shape=(environment_stepper.num_steps,)), decimals=4)
        # internal states (c- and h-state)
        self.assertTrue(out[7][0].dtype == np.float32)
        self.assertTrue(out[7][1].dtype == np.float32)
        self.assertTrue(out[7][0].shape == (environment_stepper.num_steps, 256))
        self.assertTrue(out[7][1].shape == (environment_stepper.num_steps, 256))

        # Check whether episode returns match single rewards (including terminal signals).
        episode_returns = 0.0
        for i in range(environment_stepper.num_steps):
            episode_returns += out[2][i]
            self.assertAlmostEqual(episode_returns, out[3][i])
            # Terminal: Reset for next step.
            if out[4][i] is np.bool_(True):
                episode_returns = 0.0

        test.terminate()

    def test_to_find_out_what_breaks_specifiable_server_start_via_thread_pools(self):
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
                type="dict-preprocessor-stack",
                preprocessors=dict(
                    # The images from the env  are divided by 255.
                    RGB_INTERLEAVED=[dict(type="divide", divisor=255)],
                    # The prev. action/reward from the env must be flattened/bumped-up-to-(1,).
                    previous_action=[dict(type="reshape", flatten=True, flatten_categories=action_space.num_categories)],
                    previous_reward=[dict(type="reshape", new_shape=(1,)), dict(type="convert_type", to_dtype="float32")],
                )
            ),
            # Policy spec.
            dict(network_spec=LargeIMPALANetwork(), action_space=action_space),
            # Exploration spec.
            Exploration(epsilon_spec=dict(decay_spec=dict(
                type="linear_decay", from_=1.0, to_=0.1, start_timestep=0, num_timesteps=100)
            ))
        )
        environment_stepper = EnvironmentStepper(
            environment_spec=env_spec,
            actor_component_spec=actor_component,
            state_space=state_space,
            reward_space="float32",
            internal_states_space=self.internal_states_space,
            num_steps=100,
            # Add both prev-action and -reward into the state sent through the network.
            add_previous_action=True,
            add_previous_reward=True,
            add_action_probs=True,
            action_probs_space=self.action_probs_space
        )

        test = ComponentTest(
            component=environment_stepper,
            action_space=action_space,
        )
        # Reset the stepper.
        test.test("reset")

    #def test_isolated_impala_learner_agent_functionality(self):
    #    """
    #    Creates a IMPALAAgent (learner), inserts some dummy records and "learns" from them.
    #    """
    #    agent_config = config_from_path("configs/impala_agent_for_deepmind_lab_env.json")
    #    environment_spec = dict(
    #        type="deepmind-lab", level_id="lt_hallway_slope", observations=["RGB_INTERLEAVED", "INSTR"], frameskip=4
    #    )
    #    env = DeepmindLabEnv.from_spec(environment_spec)

    #    agent = IMPALAAgent.from_spec(
    #        agent_config,
    #        type="learner",
    #        architecture="small",
    #        environment_spec=environment_spec,
    #        state_space=env.state_space,
    #        action_space=env.action_space,
    #        # TODO: automate this (by lookup from NN).
    #        internal_states_space=IMPALAAgent.standard_internal_states_space,
    #    )
    #    agent.call_api_method("insert_dummy_records")
    #    agent.call_api_method("update_from_memory")
