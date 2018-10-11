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

from copy import deepcopy
import numpy as np
from six.moves import xrange as range_
import unittest

from rlgraph.agents import ApexAgent, DQNAgent
from rlgraph.components.layers import GrayScale, Multiply
from rlgraph.components.neural_networks import PreprocessorStack
from rlgraph.environments import SequentialVectorEnv
from rlgraph.spaces import *
from rlgraph.tests import ComponentTest, recursive_assert_almost_equal
from rlgraph.tests.test_util import config_from_path


class TestPreprocessorStacks(unittest.TestCase):
    """
    Tests preprocessor stacks using different backends.
    """
    batch_size = 4

    # All preprocessors
    preprocessing_spec = [
        # Remove image-resize (make crop 80 instead 160) as tf's erroneous resize implementation
        # screws up everything.
        {
            "type": "image_crop",
            "x": 0,
            "y": 25,
            "width": 160,
            "height": 160,
            "scope": "image_crop"
        },
        #{
        #    "type": "image_resize",
        #    "width": 80,
        #    "height": 80,
        #    "scope": "image_resize"
        #},
        {
            "type": "grayscale",
            "keep_rank": True,
            "scope": "grayscale"
        },
        {
            "type": "divide",
            "divisor": 255,
            "scope": "divide"
        },
        {
            "type": "sequence",
            "sequence_length": 4,
            "batch_size": batch_size,
            "add_rank": False,
            "scope": "sequence"
        }
    ]

    preprocessing_spec_ray_pong = [
        {
            "type": "image_resize",
            "width": 84,
            "height": 84,
            "scope": "image_resize"
        },
        {
            "type": "grayscale",
            "keep_rank": True,
            "scope": "grayscale"
        },
        {
            "type": "sequence",
            "sequence_length": 4,
            "batch_size": 1,
            "add_rank": False,
            "scope": "sequence"
        }
    ]

    # TODO: Make tests backend independent so we can use the same tests for everything.
    def test_backend_equivalence(self):
        """
        Tests if Python and TensorFlow backend return the same output
        for a standard DQN-style preprocessing stack.
        """
        in_space = IntBox(256, shape=(210, 160, 3), dtype="uint8", add_batch_rank=True)

        # Regression test: Incrementally add preprocessors.
        to_use = []
        for i, decimals in zip(range_(len(self.preprocessing_spec)), [0, 0, 2, 2]):
            to_use.append(i)
            incremental_spec = []
            incremental_scopes = []
            for index in to_use:
                incremental_spec.append(deepcopy(self.preprocessing_spec[index]))
                incremental_scopes.append(self.preprocessing_spec[index]["scope"])

            print("Comparing incremental spec: {}".format(incremental_scopes))

            # Set up python preprocessor.
            # Set backend to python.
            for spec in incremental_spec:
                spec["backend"] = "python"
            python_preprocessor = PreprocessorStack(*incremental_spec, backend="python")
            for sub_comp_scope in incremental_scopes:
                python_preprocessor.sub_components[sub_comp_scope].create_variables(
                    input_spaces=dict(preprocessing_inputs=in_space), action_space=None
                )
                python_preprocessor.sub_components[sub_comp_scope].check_input_spaces(
                    input_spaces=dict(preprocessing_inputs=in_space), action_space=None
                )
                #build_space = python_processor.sub_components[sub_comp_scope].get_preprocessed_space(build_space)
                python_preprocessor.reset()

            # To compare to tf, use an equivalent tf PreprocessorStack.
            # Switch back to tf.
            for spec in incremental_spec:
                spec["backend"] = "tf"
            tf_preprocessor = PreprocessorStack(*incremental_spec, backend="tf")

            test = ComponentTest(component=tf_preprocessor, input_spaces=dict(
                inputs=in_space
            ))

            # Generate a few states from random set points. Test if preprocessed states are almost equal
            states = in_space.sample(size=self.batch_size)
            python_preprocessed_states = python_preprocessor.preprocess(states)
            tf_preprocessed_states = test.test(("preprocess", states), expected_outputs=None)

            print("Asserting (almost) equal values:")
            for tf_state, python_state in zip(tf_preprocessed_states, python_preprocessed_states):
                recursive_assert_almost_equal(tf_state, python_state, decimals=decimals)
            print("Success comparing: {}".format(incremental_scopes))

    def test_ray_pong_preprocessor_config_in_python(self):
        in_space = IntBox(256, shape=(210, 160, 3), dtype="uint8", add_batch_rank=True)

        # Regression test: Incrementally add preprocessors.
        specs = [spec for spec in self.preprocessing_spec_ray_pong]
        scopes = [spec["scope"] for spec in self.preprocessing_spec_ray_pong]

        # Set up python preprocessor.
        # Set backend to python.
        for spec in specs:
            spec["backend"] = "python"
        python_preprocessor = PreprocessorStack(*specs, backend="python")
        for sub_comp_scope in scopes:
            python_preprocessor.sub_components[sub_comp_scope].create_variables(
                input_spaces=dict(preprocessing_inputs=in_space), action_space=None
            )
            python_preprocessor.reset()

        # Generate a few states from random set points. Test if preprocessed states are almost equal
        states = in_space.sample(size=self.batch_size)
        python_preprocessed_states = python_preprocessor.preprocess(states)
        # TODO: add more checks here besides the shape.
        self.assertEqual(python_preprocessed_states.shape, (4, 84, 84, 4))

    def test_batched_backend_equivalence(self):
        return
        """
        Tests if Python and TensorFlow backend return the same output
        for a standard DQN-style preprocessing stack.
        """
        env_spec = dict(
            type="openai",
            gym_env="Pong-v0",
            frameskip=4,
            max_num_noops=30,
            episodic_life=True
        )
        # Test with batching because we assume vector environments to be the normal case going forward.
        env = SequentialVectorEnv(num_envs=4, env_spec=env_spec, num_background_envs=2)
        in_space = env.state_space

        agent_config = config_from_path("configs/ray_apex_for_pong.json")
        preprocessing_spec = deepcopy(agent_config["preprocessing_spec"])

        # Set up python preprocessor.
        scopes = [preprocessor["scope"] for preprocessor in preprocessing_spec]
        # Set backend to python.
        for spec in preprocessing_spec:
            spec["backend"] = "python"
        python_processor = PreprocessorStack(*preprocessing_spec, backend="python")
        for sub_comp_scope in scopes:
            python_processor.sub_components[sub_comp_scope].create_variables(dict(preprocessing_inputs=in_space))
        python_processor.reset()

        # To have the use case we considered so far, use agent interface for TF backend.
        agent_config.pop("type")
        agent = ApexAgent(state_space=env.state_space, action_space=env.action_space, **agent_config)

        # Generate a few states from random set points. Test if preprocessed states are almost equal
        states = np.asarray(env.reset_all())
        actions, agent_preprocessed_states = agent.get_action(
            states=states, use_exploration=False, extra_returns="preprocessed_states")
        print("TensorFlow preprocessed shape: {}".format(np.asarray(agent_preprocessed_states).shape))
        python_preprocessed_states = python_processor.preprocess(states)
        print("Python preprocessed shape: {}".format(np.asarray(python_preprocessed_states).shape))
        print("Asserting (almost) equal values:")
        for tf_state, python_state in zip(agent_preprocessed_states, python_preprocessed_states):
            flat_tf = np.ndarray.flatten(tf_state)
            flat_python = np.ndarray.flatten(python_state)
            for x, y in zip(flat_tf, flat_python):
                recursive_assert_almost_equal(x, y, decimals=3)

        states, _, _, _ = env.step(actions)
        actions, agent_preprocessed_states = agent.get_action(
            states=states, use_exploration=False, extra_returns="preprocessed_states")
        print("TensorFlow preprocessed shape: {}".format(np.asarray(agent_preprocessed_states).shape))
        python_preprocessed_states = python_processor.preprocess(states)
        print("Python preprocessed shape: {}".format(np.asarray(python_preprocessed_states).shape))
        print("Asserting (almost) equal values:")
        recursive_assert_almost_equal(agent_preprocessed_states, python_preprocessed_states, decimals=3)

    def test_simple_preprocessor_stack_with_one_preprocess_layer(self):
        stack = PreprocessorStack(dict(type="multiply", factor=0.5))

        test = ComponentTest(component=stack, input_spaces=dict(inputs=float))

        test.test("reset")
        test.test(("preprocess", 2.0), expected_outputs=1.0)

    # TODO: Make it irrelevent whether we test a python or a tf Component (API and handling should be 100% identical)
    def test_simple_python_preprocessor_stack(self):
        """
        Tests a pure python preprocessor stack.
        """
        space = FloatBox(shape=(2,), add_batch_rank=True)
        # python PreprocessorStack
        multiply = dict(type="multiply", factor=0.5, scope="m")
        divide = dict(type="divide", divisor=0.5, scope="d")
        stack = PreprocessorStack(multiply, divide, backend="python")
        for sub_comp_scope in ["m", "d"]:
            stack.sub_components[sub_comp_scope].create_variables(input_spaces=dict(inputs=space))

        #test = ComponentTest(component=stack, input_spaces=dict(inputs=float))

        for _ in range_(3):
            # Call fake API-method directly (ok for PreprocessorStack).
            stack.reset()
            input_ = np.asarray([[1.0], [2.0], [3.0], [4.0]])
            expected = input_
            #test.test(("preprocess", input_), expected_outputs=expected)
            out = stack.preprocess(input_)
            recursive_assert_almost_equal(out, input_)

            input_ = space.sample()
            #test.test(("preprocess", input_), expected_outputs=expected)
            out = stack.preprocess(input_)
            recursive_assert_almost_equal(out, input_)

    def test_preprocessor_from_list_spec(self):
        space = FloatBox(shape=(2,))
        stack = PreprocessorStack.from_spec([
            dict(type="grayscale", keep_rank=False, weights=(0.5, 0.5)),
            dict(type="divide", divisor=2),
        ])
        test = ComponentTest(component=stack, input_spaces=dict(inputs=space))

        # Run the test.
        input_ = np.array([3.0, 5.0])
        expected = np.array(2.0)
        test.test("reset")
        test.test(("preprocess", input_), expected_outputs=expected)

    def test_two_preprocessor_layers_in_a_preprocessor_stack(self):
        space = Dict(
            a=FloatBox(shape=(1, 2)),
            b=FloatBox(shape=(2, 2, 2)),
            c=Tuple(FloatBox(shape=(2,)), Dict(ca=FloatBox(shape=(3, 3, 2))))
        )

        # Construct the Component to test (PreprocessorStack).
        scale = Multiply(factor=2)
        gray = GrayScale(weights=(0.5, 0.5), keep_rank=False)
        stack = PreprocessorStack(scale, gray)
        test = ComponentTest(component=stack, input_spaces=dict(inputs=space))

        input_ = dict(
            a=np.array([[3.0, 5.0]]),
            b=np.array([[[2.0, 4.0], [2.0, 4.0]], [[2.0, 4.0], [2.0, 4.0]]]),
            c=(np.array([10.0, 20.0]), dict(ca=np.array([[[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]],
                                                         [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]],
                                                         [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]])))
        )
        expected = dict(
            a=np.array([8.0]),
            b=np.array([[6.0, 6.0], [6.0, 6.0]]),
            c=(30.0, dict(ca=np.array([[3.0, 3.0, 3.0],
                                       [3.0, 3.0, 3.0],
                                       [3.0, 3.0, 3.0]])))
        )
        test.test("reset")
        test.test(("preprocess", input_), expected_outputs=expected)
