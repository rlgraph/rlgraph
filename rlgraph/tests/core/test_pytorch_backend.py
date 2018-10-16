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
import time

from rlgraph import get_backend
from rlgraph.agents import DQNAgent, ApexAgent
from rlgraph.components import Policy, MemPrioritizedReplay
from rlgraph.environments import OpenAIGymEnv
from rlgraph.spaces import FloatBox, IntBox, Dict, BoolBox
from rlgraph.tests import ComponentTest
from rlgraph.tests.test_util import config_from_path
from rlgraph.utils import root_logger, softmax
from rlgraph.tests.dummy_components import *
from rlgraph.utils.execution_util import print_call_chain


class TestPytorchBackend(unittest.TestCase):
    """
    Tests PyTorch component execution.

    # TODO: This is a temporary test. We will later run all backend-specific
    tests via setting the executor in the component-test.
    """
    root_logger.setLevel(level=logging.INFO)

    def test_api_call_no_variables(self):
        """
        Tests define-by-run call of api method via defined_api method on a
        component without variables.
        """
        a = Dummy2To1()
        test = ComponentTest(component=a, input_spaces=dict(input1=float, input2=float))
        test.test(("run", [1.0, 2.0]), expected_outputs=3.0, decimals=4)

    def test_connecting_1to2_to_2to1(self):
        """
        Adds two components with 1-to-2 and 2-to-1 graph_fns to the core, connects them and passes a value through it.
        """
        core = Component(scope="container")
        sub_comp1 = Dummy1To2(scope="comp1")  # outs=in,in+1
        sub_comp2 = Dummy2To1(scope="comp2")  # out =in1+in2
        core.add_components(sub_comp1, sub_comp2)

        @rlgraph_api(component=core)
        def run(self_, input_):
            out1, out2 = sub_comp1.run(input_)
            return sub_comp2.run(out1, out2)

        test = ComponentTest(component=core, input_spaces=dict(input_=float))

        # Expected output: input + (input + 1.0)
        test.test(("run", 100.9), expected_outputs=np.array(202.8, dtype=np.float32))
        test.test(("run", -5.1), expected_outputs=np.array(-9.2, dtype=np.float32))

    def test_calling_sub_components_api_from_within_graph_fn(self):
        a = DummyCallingSubComponentsAPIFromWithinGraphFn(scope="A")
        test = ComponentTest(component=a, input_spaces=dict(input_=float))

        # Expected: (1): 2*in + 10
        test.test(("run", 1.1), expected_outputs=12.2, decimals=4)

    def test_1to1_to_2to1_component_with_constant_input_value(self):
        """
        Adds two components in sequence, 1-to-1 and 2-to-1, to the core and blocks one of the api_methods of 2-to-1
        with a constant value (so that this constant value is not at the border of the root-component).
        """
        core = Component(scope="container")
        sub_comp1 = Dummy1To1(scope="A")
        sub_comp2 = Dummy2To1(scope="B")
        core.add_components(sub_comp1, sub_comp2)

        @rlgraph_api(component=core)
        def run(self_, input_):
            out = sub_comp1.run(input_)
            return sub_comp2.run(out, 1.1)

        test = ComponentTest(component=core, input_spaces=dict(input_=float))

        # Expected output: (input + 1.0) + 1.1
        test.test(("run", 78.4), expected_outputs=80.5)
        test.test(("run", -5.2), expected_outputs=-3.1)

    def test_dqn_compilation(self):
        """
        Creates a DQNAgent and runs it via a Runner on an openAI Pong Env.
        """
        env = OpenAIGymEnv("Pong-v0", frameskip=4, max_num_noops=30, episodic_life=True)
        agent_config = config_from_path("configs/dqn_pytorch_test.json")
        agent = DQNAgent.from_spec(
            # Uses 2015 DQN parameters as closely as possible.
            agent_config,
            state_space=env.state_space,
            # Try with "reduced" action space (actually only 3 actions, up, down, no-op)
            action_space=env.action_space
        )

    def test_memory_compilation(self):
        # Builds a memory and returns build stats.
        env = OpenAIGymEnv("Pong-v0", frameskip=4, max_num_noops=30, episodic_life=True)

        record_space = Dict(
            states=env.state_space,
            actions=env.action_space,
            rewards=float,
            terminals=BoolBox(),
            add_batch_rank=True
        )
        input_spaces = dict(
            # insert: records
            records=record_space,
            # get_records: num_records
            num_records=int,
            # update_records: indices, update
            indices=IntBox(add_batch_rank=True),
            update=FloatBox(add_batch_rank=True)
        )

        input_spaces.pop("num_records")
        memory = MemPrioritizedReplay(
            capacity=20000,
        )
        test = ComponentTest(component=memory, input_spaces=input_spaces, auto_build=False)
        return test.build()

    # TODO -> batch dim works differently in pytorch -> have to squeeze.
    def test_dense_layer(self):
        # Space must contain batch dimension (otherwise, NNLayer will complain).
        space = FloatBox(shape=(2,), add_batch_rank=True)

        # - fixed 1.0 weights, no biases
        dense_layer = DenseLayer(units=2, weights_spec=1.0, biases_spec=False)
        test = ComponentTest(component=dense_layer, input_spaces=dict(inputs=space))

        # Batch of size=1 (can increase this to any larger number).
        input_ = np.array([0.5, 2.0])
        expected = np.array([2.5, 2.5])
        test.test(("apply", input_), expected_outputs=expected)

    def test_nn_assembly_from_file(self):
        # Space must contain batch dimension (otherwise, NNlayer will complain).
        space = FloatBox(shape=(3,), add_batch_rank=True)

        # Create a simple neural net from json.
        neural_net = NeuralNetwork.from_spec(config_from_path("configs/test_simple_nn.json"))  # type: NeuralNetwork

        # Do not seed, we calculate expectations manually.
        test = ComponentTest(component=neural_net, input_spaces=dict(inputs=space), seed=None)

        # Batch of size=3.
        input_ = np.array([[0.1, 0.2, 0.3], [1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])

        # Cant fetch variables here.

        out = test.test(("apply", input_), decimals=5)
        print(out)

    def test_policy_for_discrete_action_space(self):
        # state_space (NN is a simple single fc-layer relu network (2 units), random biases, random weights).
        state_space = FloatBox(shape=(4,), add_batch_rank=True)

        # action_space (5 possible actions).
        action_space = IntBox(5, add_batch_rank=True)

        policy = Policy(network_spec=config_from_path("configs/test_simple_nn.json"), action_space=action_space)
        test = ComponentTest(
            component=policy,
            input_spaces=dict(nn_input=state_space),
            action_space=action_space
        )
        policy_params = test.read_variable_values(policy.variables)

        # Some NN inputs (4 input nodes, batch size=2).
        states = np.array([[-0.08, 0.4, -0.05, -0.55], [13.0, -14.0, 10.0, -16.0]])
        # Raw NN-output.
        expected_nn_output = np.matmul(states, policy_params["policy/test-network/hidden-layer/dense/kernel"])
        test.test(("get_nn_output", states), expected_outputs=expected_nn_output, decimals=6)

        # Raw action layer output; Expected shape=(2,5): 2=batch, 5=action categories
        expected_action_layer_output = np.matmul(
            expected_nn_output, policy_params["policy/action-adapter/action-layer/dense/kernel"]
        )
        expected_action_layer_output = np.reshape(expected_action_layer_output, newshape=(2, 5))
        test.test(("get_action_layer_output", states), expected_outputs=expected_action_layer_output,
                  decimals=5)

        expected_actions = np.argmax(expected_action_layer_output, axis=-1)
        test.test(("get_action", states), expected_outputs=expected_actions)

        # Logits, parameters (probs) and skip log-probs (numerically unstable for small probs).
        expected_probabilities_output = softmax(expected_action_layer_output, axis=-1)
        test.test(("get_logits_probabilities_log_probs", states, [0, 1]), expected_outputs=[
            expected_action_layer_output,
            np.array(expected_probabilities_output, dtype=np.float32)
            # np.log(expected_probabilities_output)
        ], decimals=5)

        print("Probs: {}".format(expected_probabilities_output))

        # Stochastic sample.
        expected_actions = np.array([3, 4])
        test.test(("get_stochastic_action", states), expected_outputs=expected_actions)

        # Deterministic sample.
        expected_actions = np.array([4, 4])
        test.test(("get_max_likelihood_action", states), expected_outputs=expected_actions)

        # Distribution's entropy.
        expected_h = np.array([1.572, 0.003])
        test.test(("get_entropy", states), expected_outputs=expected_h, decimals=3)

    def test_act(self):
        env = OpenAIGymEnv("Pong-v0", frameskip=4, max_num_noops=30, episodic_life=True)
        agent_config = config_from_path("configs/ray_apex_for_pong.json")
        if get_backend() == "pytorch":
            agent_config["memory_spec"]["type"] = "mem_prioritized_replay"
        agent = DQNAgent.from_spec(
            # Uses 2015 DQN parameters as closely as possible.
            agent_config,
            state_space=env.state_space,
            # Try with "reduced" action space (actually only 3 actions, up, down, no-op)
            action_space=env.action_space
        )
        state = env.reset()
        action = agent.get_action(state)
        print("Component call count = {}".format(Component.call_count))

        state_space = env.state_space
        count = 200

        samples = state_space.sample(count)
        start = time.perf_counter()
        for s in samples:
            action = agent.get_action(s)
        end = time.perf_counter() - start

        print("Took {} s for {} separate actions, mean = {}".format(end, count, end / count))

        # Now instead test 100 batch actions
        samples = state_space.sample(count)
        start = time.perf_counter()
        action = agent.get_action(samples)
        end = time.perf_counter() - start
        print("Took {} s for {} batched actions.".format(end, count))
        profile = Component.call_times
        print_call_chain(profile, False, 0.03)

    def test_get_td_loss(self):
        env = OpenAIGymEnv("Pong-v0", frameskip=4, max_num_noops=30, episodic_life=True)
        agent_config = config_from_path("configs/ray_apex_for_pong.json")

        # Test cpu settings for batching here.
        agent_config["memory_spec"]["type"] = "mem_prioritized_replay"
        agent_config["execution_spec"]["torch_num_threads"] = 1
        agent_config["execution_spec"]["OMP_NUM_THREADS"] = 1

        agent = ApexAgent.from_spec(
            # Uses 2015 DQN parameters as closely as possible.
            agent_config,
            state_space=env.state_space,
            # Try with "reduced" action space (actually only 3 actions, up, down, no-op)
            action_space=env.action_space
        )
        samples = 200
        rewards = np.random.random(size=samples)
        states = list(agent.preprocessed_state_space.sample(samples))
        actions = agent.action_space.sample(samples)
        terminals = np.zeros(samples, dtype=np.uint8)
        next_states = states[1:]
        next_states.extend([agent.preprocessed_state_space.sample(1)])
        next_states = np.asarray(next_states)
        states = np.asarray(states)
        weights = np.ones_like(rewards)

        for _ in range(1):
            start = time.perf_counter()
            _, loss_per_item = agent.get_td_loss(
                dict(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    terminals=terminals,
                    next_states=next_states,
                    importance_weights=weights
                )
            )
            print("post process time = {}".format(time.perf_counter() - start))
        profile = Component.call_times
        print_call_chain(profile, False, 0.003)
