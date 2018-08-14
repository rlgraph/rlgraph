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
from six.moves import xrange as range_
import unittest

from rlgraph.components.component import Component
from rlgraph.components.action_adapters.action_adapter import ActionAdapter
from rlgraph.components.explorations.exploration import Exploration, EpsilonExploration
from rlgraph.components.common.decay_components import LinearDecay
from rlgraph.components.distributions import Categorical, Normal
from rlgraph.spaces import *
from rlgraph.tests import ComponentTest


class TestExplorations(unittest.TestCase):

    def test_epsilon_exploration(self):
        # TODO not portable, redo.
        return
        # Decaying a value always without batch dimension (does not make sense for global time step).
        time_step_space = IntBox(add_batch_rank=False)
        sample_space = FloatBox(add_batch_rank=True)

        # The Component(s) to test.
        decay_component = LinearDecay(from_=1.0, to_=0.0, start_timestep=0, num_timesteps=1000)
        epsilon_component = EpsilonExploration(decay_spec=decay_component)
        test = ComponentTest(component=epsilon_component, input_spaces=dict(sample=sample_space,
                                                                            time_step=time_step_space))

        # Values to pass as single items.
        input_ = np.array([0, 1, 2, 25, 50, 100, 110, 112, 120, 130, 150, 180, 190, 195, 200, 201, 210, 250, 386,
                           670, 789, 900, 923, 465, 894, 91, 1000])
        expected = np.array([True, True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                             True, True, False, True, True, False, False, False, False, False, True, False])
        for i, e in zip(input_, expected):
            # Only pass in sample (zeros) for the batch rank.
            test.test(("do_explore", [np.zeros(shape=(1,)), i]), expected_outputs=e)

    def test_exploration_with_discrete_action_space(self):
        # TODO not portable, redo.
        return
        # 2x2 action-pick, each composite action with 5 categories.
        action_space = IntBox(5, shape=(2, 2), add_batch_rank=True)
        # Our distribution to go into the Exploration object.
        distribution = Categorical()
        action_adapter = ActionAdapter(action_space=action_space)
        nn_output_space = FloatBox(shape=(13,), add_batch_rank=True)  # 13: Any flat nn-output should be ok.
        exploration = Exploration.from_spec(dict(
            epsilon_spec=dict(
                decay_spec=dict(
                    type="linear_decay",
                    from_=1.0,
                    to_=0.1,
                    start_timestep=0,
                    num_timesteps=10000
                )
            )
        ))
        # The Component to test.
        exploration_pipeline = Component(action_adapter, distribution, exploration, scope="exploration-pipeline")

        def get_action(self_, nn_output, time_step):
            _, parameters, _ = self_.call(action_adapter.get_logits_parameters_log_probs, nn_output)
            sample_stochastic = self_.call(distribution.sample_stochastic, parameters)
            sample_deterministic = self_.call(distribution.sample_deterministic, parameters)
            action = self_.call(exploration.get_action, sample_stochastic, sample_deterministic, time_step)
            return action

        exploration_pipeline.define_api_method("get_action", get_action)

        test = ComponentTest(component=exploration_pipeline,
                             input_spaces=dict(nn_output=nn_output_space, time_step=int),
                             action_space=action_space)

        # fake output from last NN layer (shape=(13,))
        inputs = [
            np.array([[100.0, 50.0, 25.0, 12.5, 6.25,
                       200.0, 100.0, 50.0, 25.0, 12.5,
                       1.0, 1.0, 25.0
                       ],
                      [123.4, 34.7, 98.2, 1.2, 120.0,
                       200.0, 200.0, 0.00009, 10.0, 300.0,
                       0.567, 0.678, 0.789
                       ]
                      ]),
            10000
        ]
        expected = np.array([[[1, 2], [2, 4]], [[2, 1], [0, 3]]])
        test.test(("get_action", inputs), expected_outputs=expected)

    def test_exploration_with_continuous_action_space(self):
        # TODO not portable, redo.
        return
        # 2x2 action-pick, each composite action with 5 categories.
        action_space = FloatBox(shape=(2,2), add_batch_rank=True)

        distribution = Normal()
        action_adapter = ActionAdapter(action_space=action_space)

        # Our distribution to go into the Exploration object.
        nn_output_space = FloatBox(shape=(13,), add_batch_rank=True)  # 13: Any flat nn-output should be ok.

        exploration = Exploration.from_spec(dict(noise_spec=dict(type="gaussian_noise", mean=10.0, stddev=2.0)))

        # The Component to test.
        exploration_pipeline = Component(scope="continuous-plus-noise")
        exploration_pipeline.add_components(action_adapter, distribution, exploration, scope="exploration-pipeline")

        def get_action(self_, nn_output):
            _, parameters, _ = self_.call(action_adapter.get_logits_parameters_log_probs, nn_output)
            sample_stochastic = self_.call(distribution.sample_stochastic, parameters)
            sample_deterministic = self_.call(distribution.sample_deterministic, parameters)
            action = self_.call(exploration.get_action, sample_stochastic, sample_deterministic)
            return action

        exploration_pipeline.define_api_method("get_action", get_action)

        def get_noise(self_):
            return self_.call(exploration.noise_component.get_noise)

        exploration_pipeline.define_api_method("get_noise", get_noise)

        test = ComponentTest(component=exploration_pipeline,
                             input_spaces=dict(nn_output=nn_output_space),
                             action_space=action_space)

        # Collect outputs in `collected` list to compare moments.
        collected = list()
        for _ in range_(1000):
            test.test("get_noise", fn_test=lambda component_test, outs: collected.append(outs))

        self.assertAlmostEqual(10.0, np.mean(collected), places=1)
        self.assertAlmostEqual(2.0, np.std(collected), places=1)

        np.random.seed(10)
        input_ = nn_output_space.sample(size=3)
        expected = np.array([[[13.163095, 8.46925],
                              [10.375976, 5.4675055]],
                             [[13.239931, 7.990649],
                              [10.03761, 10.465796]],
                             [[10.280741, 7.2384844],
                              [10.040194, 8.248206]]], dtype=np.float32)
        test.test(("get_action", input_), expected_outputs=expected, decimals=3)
