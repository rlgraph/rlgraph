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

from rlgraph.components import Component, ActionAdapter, Exploration, EpsilonExploration, LinearDecay
from rlgraph.components.distributions import Categorical, Normal
from rlgraph.spaces import *
from rlgraph.tests import ComponentTest


class TestExplorations(unittest.TestCase):

    def test_epsilon_exploration(self):
        # Decaying a value always without batch dimension (does not make sense for global time step).
        time_step_space = IntBox(add_batch_rank=False)

        # The Component(s) to test.
        decay_component = LinearDecay(from_=1.0, to_=0.0, start_timestep=0, num_timesteps=1000)
        epsilon_component = EpsilonExploration(decay_spec=decay_component)
        test = ComponentTest(component=epsilon_component, input_spaces=dict(do_explore=time_step_space))

        # Values to pass as single items.
        input_ = np.array([0, 1, 2, 25, 50, 100, 110, 112, 120, 130, 150, 180, 190, 195, 200, 201, 210, 250, 386,
                           670, 789, 900, 923, 465, 894, 91, 1000])
        expected = np.array([True, True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                             True, True, False, True, True, False, False, False, False, False, True, False])
        for i, e in zip(input_, expected):
            test.test(("do_explore", i), expected_outputs=e)

    def test_exploration_with_discrete_action_space(self):
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
                             input_spaces=dict(get_action=[nn_output_space, int]),
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
        expected = np.array([[[3, 1], [3, 2]], [[1, 1], [3, 2]]])
        test.test(("get_action", inputs), expected_outputs=expected)

    def test_exploration_with_continuous_action_space(self):
        # 2x2 action-pick, each composite action with 5 categories.
        action_space = FloatBox(shape=(2,2), add_batch_rank=True)

        distribution = Normal()
        action_adapter = ActionAdapter(action_space=action_space)

        # Our distribution to go into the Exploration object.
        nn_output_space = FloatBox(shape=(13,), add_batch_rank=True)  # 13: Any flat nn-output should be ok.

        exploration = Exploration.from_spec(dict(noise_spec=dict(type="gaussian_noise", mean=10.0, sd=2.0)))

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
                             input_spaces=dict(get_action=nn_output_space),
                             action_space=action_space)

        # Collect outputs in `collected` list to compare moments.
        collected = list()

        def collect_outs(component_test, outs):
            return collected.append(outs)

        for i in range_(1000):
            test.test("get_noise", fn_test=collect_outs)

        self.assertAlmostEqual(10.0, np.mean(collected), places=1)
        self.assertAlmostEqual(2.0, np.std(collected), places=1)

        np.random.seed(10)
        input_ = nn_output_space.sample(size=3)
        expected = np.array([
            [
                [14.051712, 8.483587],
                [10.151371, 7.607021]
            ],
            [
                [13.607159, 8.198939],
                [10.093333, 7.73495]
            ],
            [
                [13.657397, 9.592756],
                [9.908859, 6.4407268]
            ]
        ], dtype=np.float32)
        test.test(("get_action", input_), expected_outputs=expected)
