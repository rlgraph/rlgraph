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

import numpy as np
import unittest

from rlgraph.components.neural_networks.actor_component import ActorComponent
from rlgraph.components.explorations.exploration import Exploration
from rlgraph.components.neural_networks.policy import Policy
from rlgraph.components.neural_networks.preprocessor_stack import PreprocessorStack
from rlgraph.spaces import *
from rlgraph.tests import ComponentTest
from rlgraph.tests.test_util import config_from_path
from rlgraph.utils import softmax, relu


class TestActorComponents(unittest.TestCase):

    def test_simple_actor_component(self):
        # state_space (NN is a simple single fc-layer relu network (2 units), random biases, random weights).
        state_space = FloatBox(shape=(5,), add_batch_rank=True)
        # action_space.
        action_space = IntBox(10, add_batch_rank=True)

        preprocessor = PreprocessorStack.from_spec(
            [dict(type="convert_type", to_dtype="float"), dict(type="multiply", factor=2)]
        )
        policy = Policy(neural_network=config_from_path("configs/test_simple_nn.json"), action_space=action_space)
        exploration = Exploration()  # no exploration
        actor_component = ActorComponent(preprocessor, policy, exploration)
        test = ComponentTest(
            component=actor_component,
            input_spaces=dict(states=state_space),
            action_space=action_space
        )
        # Some state inputs (5 input nodes, batch size=2).
        states = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        # Get and check some actions.
        actor_component_params = test.read_variable_values(actor_component.variables)
        # Expected NN-output.
        expected_nn_output = np.matmul(
            states * 2, actor_component_params["actor-component/policy/test-network/hidden-layer/dense/kernel"]
        )
        # Raw action layer output.
        expected_action_layer_output = np.matmul(
            expected_nn_output,
            actor_component_params["actor-component/policy/action-adapter/action-layer/dense/kernel"]
        )
        # Final actions (max-likelihood/greedy pick).
        expected_actions = np.argmax(expected_action_layer_output, axis=-1)
        expected_preprocessed_state = states * 2
        test.test(
            ("get_preprocessed_state_and_action", states),
            expected_outputs=(expected_preprocessed_state, expected_actions)
        )
