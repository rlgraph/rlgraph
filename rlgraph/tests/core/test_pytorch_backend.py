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

from rlgraph import spaces
from rlgraph.environments import Environment
from rlgraph.spaces import FloatBox
from rlgraph.tests import ComponentTest
from rlgraph.tests.test_util import config_from_path
from rlgraph.utils import root_logger
from rlgraph.tests.dummy_components import *


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

        def run(self_, input_):
            out1, out2 = self_.call(sub_comp1.run, input_)
            return self_.call(sub_comp2.run, out1, out2)

        core.define_api_method("run", run)

        test = ComponentTest(component=core, input_spaces=dict(input_=float))

        # Expected output: input + (input + 1.0)
        test.test(("run", 100.9), expected_outputs=202.8)
        test.test(("run", -5.1), expected_outputs=-9.2)

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

        def run(self_, input_):
            out = self_.call(sub_comp1.run, input_)
            return self_.call(sub_comp2.run, out, 1.1)

        core.define_api_method("run", run)

        test = ComponentTest(component=core, input_spaces=dict(input_=float))

        # Expected output: (input + 1.0) + 1.1
        test.test(("run", 78.4), expected_outputs=80.5)
        test.test(("run", -5.2), expected_outputs=-3.1)

    # TODO delete after debugging.
    def test_layer_passes(self):
        import torch
        torch.set_default_tensor_type("torch.FloatTensor")
        env_spec = dict(
            type="openai",
            gym_env="CartPole-v0",
            fire_reset=False
        )
        env = Environment.from_spec(env_spec)
        space = env.state_space
        print(space.shape)

        # This works
        self.fc1 = torch.nn.Linear(space.get_shape()[0], 1, bias=False)

        # Test single forward pass.
        space_sample = space.sample()
        print(space_sample)
        from_numpy_in = torch.tensor(space_sample, dtype=torch.float32, requires_grad=False)
        print(self.fc1(from_numpy_in))

        # Test batch forward pass
        space_sample = space.sample(size=10)
        from_numpy_in = torch.tensor(space_sample, dtype=torch.float32, requires_grad=False)
        print(self.fc1(from_numpy_in))

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

    def test_2_containers_flattening_splitting(self):
        """
        Adds a single component with 2-to-2 graph_fn to the core and passes two containers through it
        with flatten/split options enabled.
        """
        input1_space = spaces.Dict(a=float, b=spaces.FloatBox(shape=(1, 2)))
        input2_space = spaces.Dict(a=float, b=float)

        component = FlattenSplitDummy()
        test = ComponentTest(
            component=component,
            input_spaces=dict(input1=input1_space, input2=input2_space)
        )

        # Options: fsu=flat/split/un-flat.
        in1_fsu = dict(a=np.array(0.234), b=np.array([[0.0, 3.0]]))
        in2_fsu = dict(a=np.array(5.0), b=np.array(5.5))
        # Result of sending 'a' keys through graph_fn: (in1[a]+1.0=1.234, in1[a]+in2[a]=5.234)
        # Result of sending 'b' keys through graph_fn: (in1[b]+1.0=[[1, 4]], in1[b]+in2[b]=[[5.5, 8.5]])
        out1_fsu = dict(a=1.234, b=np.array([[1.0, 4.0]]))
        out2_fsu = dict(a=np.array(5.234, dtype=np.float32), b=np.array([[5.5, 8.5]]))
        test.test(("run", [in1_fsu, in2_fsu]), expected_outputs=[out1_fsu, out2_fsu])