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

import logging
import numpy as np
import time
import unittest

from rlgraph.agents.impala_agents import IMPALAAgent, SingleIMPALAAgent
from rlgraph.components.common.environment_stepper import EnvironmentStepper
from rlgraph.components.policies.shared_value_function_policy import SharedValueFunctionPolicy
from rlgraph.components.neural_networks.actor_component import ActorComponent
from rlgraph.components.neural_networks.impala.impala_networks import LargeIMPALANetwork
from rlgraph.environments import Environment
from rlgraph.spaces import *
from rlgraph.utils.ops import DataOpTuple
from rlgraph.tests.component_test import ComponentTest
from rlgraph.tests.test_util import recursive_assert_almost_equal, config_from_path
from rlgraph.utils import root_logger, default_dict


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
    parameters_and_logits_space = FloatBox(shape=(9,), add_batch_rank=True)
    internal_states_space = Tuple(FloatBox(shape=(256,)), FloatBox(shape=(256,)), add_batch_rank=True)
    cluster_spec = dict(learner=["localhost:22222"], actor=["localhost:22223"])
    cluster_spec_single_actor = dict(actor=["localhost:22223"])

    def test_large_impala_network_without_agent(self):
        """
        Creates a large IMPALA architecture network and runs an input sample through it.
        """
        # Create the network (with a small time-step value for this test).
        large_impala_architecture = LargeIMPALANetwork(worker_sample_size=2)
        test = ComponentTest(
            large_impala_architecture,
            input_spaces=dict(input_dict=self.input_space),
            execution_spec=dict(disable_monitoring=True)
        )

        # Send a 2x3 sample through the network (2=sequence-length (time-rank), 3=batch-size).
        sample_input = self.input_space.sample(size=(2, 3))
        expected = None
        ret = test.test(("apply", sample_input), expected_outputs=expected)
        # Check shape of outputs.
        self.assertEquals(ret["output"].shape, (2, 3, 256))
        # Check shapes of current internal_states (c and h).
        self.assertEquals(ret["last_internal_states"][0].shape, (3, 256))
        self.assertEquals(ret["last_internal_states"][1].shape, (3, 256))

        test.terminate()

    def test_large_impala_policy_without_agent(self):
        """
        Creates a large IMPALA architecture network inside a policy and runs a few input samples through it.
        """
        # Create the network.
        large_impala_architecture = LargeIMPALANetwork(worker_sample_size=1)
        # IMPALA uses a baseline action adapter (v-trace off-policy PG with baseline value function).
        policy = SharedValueFunctionPolicy(
            network_spec=large_impala_architecture, action_space=self.action_space,
            switched_off_apis={
                #"get_action_from_logits_and_parameters", "get_action_from_logits_and_probabilities",
                "get_action_log_probs"
            }
        )
        test = ComponentTest(
            policy,
            input_spaces=dict(
                nn_input=self.input_space,
                internal_states=self.internal_states_space,
                #parameters=self.parameters_and_logits_space,
                #logits=self.parameters_and_logits_space
            ),
            action_space=self.action_space,
            execution_spec=dict(disable_monitoring=True)
        )

        # Send a 1x1 sample through the network (1=sequence-length (time-rank), 1=batch-size).
        nn_input = self.input_space.sample(size=(1, 1))
        initial_internal_states = self.internal_states_space.zeros(size=1)
        expected = None
        out = test.test(("get_action", [nn_input, initial_internal_states]), expected_outputs=expected)
        print("First action: {}".format(out["action"]))
        self.assertEquals(out["action"].shape, (1, 1))
        self.assertEquals(out["last_internal_states"][0].shape, (1, 256))
        self.assertEquals(out["last_internal_states"][1].shape, (1, 256))

        # Send another 1x1 sample through the network using the previous internal-state.
        next_nn_input = self.input_space.sample(size=(1, 1))
        expected = None
        out = test.test(("get_action", [next_nn_input, out["last_internal_states"]]), expected_outputs=expected)
        print("Second action: {}".format(out["action"]))
        self.assertEquals(out["action"].shape, (1, 1))
        self.assertEquals(out["last_internal_states"][0].shape, (1, 256))
        self.assertEquals(out["last_internal_states"][1].shape, (1, 256))

        test.terminate()

    def test_large_impala_actor_component_without_agent(self):
        """
        Creates a large IMPALA architecture network inside a policy inside an actor component and runs a few input
        samples through it.
        """
        batch_size = 4
        time_steps = 1

        # IMPALA uses a baseline action adapter (v-trace off-policy PG with baseline value function).
        policy = SharedValueFunctionPolicy(LargeIMPALANetwork(worker_sample_size=time_steps),
                                           action_space=self.action_space, deterministic=False)
        actor_component = ActorComponent(preprocessor_spec=None, policy_spec=policy, exploration_spec=None)

        test = ComponentTest(
            actor_component, input_spaces=dict(
                states=self.input_space,
                internal_states=self.internal_states_space
            ),
            action_space=self.action_space,
            execution_spec=dict(disable_monitoring=True)
        )

        # Send a sample through the network (sequence-length (time-rank) x batch-size).
        nn_dict_input = self.input_space.sample(size=(time_steps, batch_size))
        initial_internal_states = self.internal_states_space.zeros(size=batch_size)
        expected = None
        out = test.test(
            ("get_preprocessed_state_and_action", [nn_dict_input, initial_internal_states]), expected_outputs=expected
        )
        print("First action: {}".format(out["action"]))
        self.assertEquals(out["action"].shape, (time_steps, batch_size))
        self.assertEquals(out["last_internal_states"][0].shape, (batch_size, 256))
        self.assertEquals(out["last_internal_states"][1].shape, (batch_size, 256))
        # Check preprocessed state (all the same except 'image' channel).
        recursive_assert_almost_equal(
            out["preprocessed_state"], dict(
                RGB_INTERLEAVED=nn_dict_input["RGB_INTERLEAVED"],
                INSTR=nn_dict_input["INSTR"],
                previous_action=nn_dict_input["previous_action"],
                previous_reward=nn_dict_input["previous_reward"],
            )
        )

        # Send another 1x1 sample through the network using the previous internal-state.
        next_nn_input = self.input_space.sample(size=(time_steps, batch_size))
        expected = None
        out = test.test(
            ("get_preprocessed_state_and_action", [next_nn_input, out["last_internal_states"]]), expected_outputs=expected
        )
        print("Second action: {}".format(out["action"]))
        self.assertEquals(out["action"].shape, (time_steps, batch_size))
        self.assertEquals(out["last_internal_states"][0].shape, (batch_size, 256))
        self.assertEquals(out["last_internal_states"][1].shape, (batch_size, 256))
        # Check preprocessed state (all the same except 'image' channel, which gets divided by 255).
        recursive_assert_almost_equal(
            out["preprocessed_state"], dict(
                RGB_INTERLEAVED=next_nn_input["RGB_INTERLEAVED"],
                INSTR=next_nn_input["INSTR"],
                previous_action=next_nn_input["previous_action"],
                previous_reward=next_nn_input["previous_reward"],
            )
        )

        test.terminate()

    def test_environment_stepper_component_with_large_impala_architecture(self):
        try:
            from rlgraph.environments.deepmind_lab import DeepmindLabEnv
        except ImportError:
            print("DeepmindLab not installed: Skipping this test case.")
            return

        worker_sample_size = 100
        env_spec = dict(
            type="deepmind_lab", level_id="seekavoid_arena_01", observations=["RGB_INTERLEAVED", "INSTR"],
            frameskip=4
        )
        dummy_env = DeepmindLabEnv.from_spec(env_spec)
        state_space = dummy_env.state_space
        action_space = dummy_env.action_space
        actor_component = ActorComponent(
            # Preprocessor spec (only for image and prev-action channel).
            dict(
                type="dict-preprocessor-stack",
                preprocessors=dict(
                    # The prev. action/reward from the env must be flattened/bumped-up-to-(1,).
                    previous_action=[dict(type="reshape", flatten=True, flatten_categories=action_space.num_categories)],
                    previous_reward=[dict(type="reshape", new_shape=(1,)), dict(type="convert_type", to_dtype="float32")],
                )
            ),
            # Policy spec. worker_sample_size=1 as its an actor network.
            dict(network_spec=LargeIMPALANetwork(worker_sample_size=1), action_space=action_space)
        )
        environment_stepper = EnvironmentStepper(
            environment_spec=env_spec,
            actor_component_spec=actor_component,
            state_space=state_space,
            reward_space="float32",
            internal_states_space=self.internal_states_space,
            num_steps=worker_sample_size,
            # Add both prev-action and -reward into the state sent through the network.
            add_previous_action_to_state=True,
            add_previous_reward_to_state=True,
            add_action_probs=True,
            action_probs_space=self.action_probs_space
        )

        test = ComponentTest(
            component=environment_stepper,
            action_space=action_space,
            execution_spec=dict(disable_monitoring=True)
        )

        environment_stepper.environment_server.start_server()

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
        self.assertTrue(out[0].dtype == np.bool_)  # next-state is terminal?
        self.assertTrue(out[1]["INSTR"].dtype == np.object)
        self.assertTrue(out[1]["RGB_INTERLEAVED"].dtype == np.uint8)
        self.assertTrue(out[1]["RGB_INTERLEAVED"].shape == (worker_sample_size + 1,) + state_space["RGB_INTERLEAVED"].shape)
        self.assertTrue(out[1]["RGB_INTERLEAVED"].min() >= 0)  # make sure we have pixels
        self.assertTrue(out[1]["RGB_INTERLEAVED"].max() <= 255)
        self.assertTrue(out[1]["previous_action"].dtype == np.int32)  # actions
        self.assertTrue(out[1]["previous_action"].shape == (worker_sample_size + 1,))
        self.assertTrue(out[1]["previous_reward"].dtype == np.float32)  # rewards
        self.assertTrue(out[1]["previous_reward"].shape == (worker_sample_size + 1,))
        # action probs (test whether sum to one).
        self.assertTrue(out[2].dtype == np.float32)
        self.assertTrue(out[2].shape == (100, action_space.num_categories))
        self.assertTrue(out[2].min() >= 0.0)
        self.assertTrue(out[2].max() <= 1.0)
        recursive_assert_almost_equal(out[2].sum(axis=-1, keepdims=False),
                                      np.ones(shape=(worker_sample_size,)), decimals=4)
        # internal states (c- and h-state)
        self.assertTrue(out[3][0].dtype == np.float32)
        self.assertTrue(out[3][0].shape == (worker_sample_size + 1, 256))
        self.assertTrue(out[3][1].dtype == np.float32)
        self.assertTrue(out[3][1].shape == (worker_sample_size + 1, 256))

        environment_stepper.environment_server.stop_server()

        test.terminate()

    def test_single_impala_agent_functionality(self):
        """
        Creates a single IMPALAAgent and runs it for a few steps in a DeepMindLab Env to test
        all steps of the actor and learning process.
        """
        try:
            from rlgraph.environments.deepmind_lab import DeepmindLabEnv
        except ImportError:
            print("Deepmind Lab not installed: Will skip this test.")
            return

        agent_config = config_from_path("configs/impala_agent_for_deepmind_lab_env.json")
        env_spec = dict(level_id="lt_hallway_slope", observations=["RGB_INTERLEAVED", "INSTR"], frameskip=4)
        dummy_env = DeepmindLabEnv.from_spec(env_spec)

        agent = SingleIMPALAAgent.from_spec(
            default_dict(dict(type="single-impala-agent"), agent_config),
            architecture="large",
            environment_spec=default_dict(dict(type="deepmind-lab"), env_spec),
            state_space=dummy_env.state_space,
            action_space=dummy_env.action_space,
            # TODO: automate this (by lookup from NN).
            internal_states_space=IMPALAAgent.default_internal_states_space,
            # Summarize time-steps to have an overview of the env-stepping speed.
            summary_spec=dict(summary_regexp="time-step", directory="/home/rlgraph/"),
            dynamic_batching=False,
            num_workers=4
        )
        # Count items in the queue.
        print("Items in queue: {}".format(agent.call_api_method("get_queue_size")))

        updates = 5
        update_times = list()
        print("Updating from queue ...")
        for _ in range(updates):
            start_time = time.monotonic()
            agent.update()
            update_times.append(time.monotonic() - start_time)

        print("Updates per second (including waiting for enqueued items): {}/s".format(updates / np.sum(update_times)))

        time.sleep(5)

        agent.terminate()

    def test_isolated_impala_actor_agent_functionality(self):
        """
        Creates a non-distributed IMPALAAgent (actor) and runs it for a few steps in a DeepMindLab Env to test
        all steps of the learning process.
        """
        try:
            from rlgraph.environments.deepmind_lab import DeepmindLabEnv
        except ImportError:
            print("Deepmind Lab not installed: Will skip this test.")
            return

        agent_config = config_from_path("configs/impala_agent_for_deepmind_lab_env.json")
        env_spec = dict(level_id="seekavoid_arena_01", observations=["RGB_INTERLEAVED", "INSTR"], frameskip=4)
        dummy_env = DeepmindLabEnv.from_spec(env_spec)

        agent = IMPALAAgent.from_spec(
            agent_config,
            type="actor",
            architecture="large",
            environment_spec=default_dict(dict(type="deepmind-lab"), env_spec),
            state_space=dummy_env.state_space,
            action_space=dummy_env.action_space,
            # TODO: automate this (by lookup from NN).
            internal_states_space=IMPALAAgent.default_internal_states_space,
            execution_spec=dict(
                #mode="distributed",
                #distributed_spec=dict(job="actor", task_index=0, cluster_spec=self.cluster_spec_single_actor),
                disable_monitoring=True
            ),
            # Need large queue to be able to fill it up (don't have a learner).
            fifo_queue_spec=dict(capacity=10000)
        )
        # Start Specifiable Server with Env manually (monitoring is disabled).
        agent.environment_stepper.environment_server.start_server()
        time_start = time.perf_counter()
        steps = 5
        for _ in range(steps):
            agent.call_api_method("perform_n_steps_and_insert_into_fifo")
        time_total = time.perf_counter() - time_start
        print("Done running {}x{} steps in Deepmind Lab env using IMPALA network in {}sec ({} actions/sec).".format(
            steps, agent.worker_sample_size, time_total , agent.worker_sample_size * steps / time_total)
        )
        agent.environment_stepper.environment_server.stop_server()
        agent.terminate()

    #def test_distributed_impala_agent_functionality_actor_part(self):
    #    """
    #    Creates an IMPALAAgent (actor) and starts it without the learner piece.
    #    Distributed actor agents are able to run autonomously as they don't require the learner to be present
    #    and connected to the server.
    #    """
    #    try:
    #        from rlgraph.environments.deepmind_lab import DeepmindLabEnv
    #    except ImportError:
    #        print("Deepmind Lab not installed: Will skip this test.")
    #        return

    #    agent_config = config_from_path("configs/impala_agent_for_deepmind_lab_env.json")
    #    env_spec = dict(level_id="seekavoid_arena_01", observations=["RGB_INTERLEAVED", "INSTR"], frameskip=4)
    #    dummy_env = DeepmindLabEnv.from_spec(env_spec)
    #    agent = IMPALAAgent.from_spec(
    #        agent_config,
    #        type="actor",
    #        architecture="large",
    #        environment_spec=default_dict(dict(type="deepmind-lab"), env_spec),
    #        state_space=dummy_env.state_space,
    #        action_space=dummy_env.action_space,
    #        # TODO: automate this (by lookup from NN).
    #        internal_states_space=IMPALAAgent.default_internal_states_space,
    #        # Setup distributed tf.
    #        execution_spec=dict(
    #            mode="distributed",
    #            distributed_spec=dict(job="actor", task_index=0, cluster_spec=self.cluster_spec),
    #            session_config=dict(
    #                type="monitored-training-session",
    #                #log_device_placement=True
    #            ),
    #            #enable_profiler=True,
    #            #profiler_frequency=1
    #        ),
    #        fifo_queue_spec=dict(capacity=10000)
    #    )
    #    time_start = time.perf_counter()
    #    steps = 50
    #    for _ in range(steps):
    #        agent.call_api_method("perform_n_steps_and_insert_into_fifo")
    #    time_total = time.perf_counter() - time_start
    #    print("Done running {}x{} steps in Deepmind Lab env using IMPALAAgent in {}sec ({} actions/sec).".format(
    #        steps, agent.worker_sample_size, time_total, agent.worker_sample_size * steps / time_total)
    #    )
    #    agent.terminate()

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
