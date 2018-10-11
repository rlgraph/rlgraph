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
from rlgraph.components.distributions import Categorical, Normal
from rlgraph.spaces import *
from rlgraph.tests import ComponentTest
from rlgraph.utils.decorators import rlgraph_api


class TestExplorations(unittest.TestCase):

    def test_exploration_with_discrete_action_space(self):
        nn_output_space = FloatBox(shape=(13,), add_batch_rank=True)  # 13: Any flat nn-output should be ok.
        time_step_space = IntBox(10000)
        # 2x2 action-pick, each composite action with 5 categories.
        action_space = IntBox(5, shape=(2, 2), add_batch_rank=True)

        # Our distribution to go into the Exploration object.
        distribution = Categorical()
        action_adapter = ActionAdapter(action_space=action_space)

        exploration = Exploration.from_spec(dict(
            epsilon_spec=dict(
                decay_spec=dict(
                    type="linear_decay",
                    from_=1.0,
                    to_=0.0,
                    start_timestep=0,
                    num_timesteps=10000
                )
            )
        ))
        # The Component to test.
        exploration_pipeline = Component(action_adapter, distribution, exploration, scope="exploration-pipeline")

        @rlgraph_api(component=exploration_pipeline)
        def get_action(self_, nn_output, time_step):
            out = action_adapter.get_logits_probabilities_log_probs(nn_output)
            sample = distribution.sample_deterministic(out["probabilities"])
            action = exploration.get_action(sample, time_step)
            return action

        test = ComponentTest(
            component=exploration_pipeline,
            input_spaces=dict(nn_output=nn_output_space, time_step=int),
            action_space=action_space
        )

        # With exploration: Check, whether actions are equally distributed.
        nn_outputs = nn_output_space.sample(2)
        time_steps = time_step_space.sample(30)
        # Collect action-batch-of-2 for each of our various random time steps.
        # Each action is an int box of shape=(2,2)
        actions = np.ndarray(shape=(30, 2, 2, 2), dtype=np.int)
        for i, time_step in enumerate(time_steps):
            actions[i] = test.test(("get_action", [nn_outputs, time_step]), expected_outputs=None)

        # Assert some distribution of the actions.
        mean_action = actions.mean()
        stddev_action = actions.std()
        self.assertAlmostEqual(mean_action, 2.0, places=0)
        self.assertAlmostEqual(stddev_action, 1.0, places=0)

        # Without exploration (epsilon is force-set to 0.0): Check, whether actions are always the same
        # (given same nn_output all the time).
        nn_outputs = nn_output_space.sample(2)
        time_steps = time_step_space.sample(30) + 10000
        # Collect action-batch-of-2 for each of our various random time steps.
        # Each action is an int box of shape=(2,2)
        actions = np.ndarray(shape=(30, 2, 2, 2), dtype=np.int)
        for i, time_step in enumerate(time_steps):
            actions[i] = test.test(("get_action", [nn_outputs, time_step]), expected_outputs=None)

        # Assert zero stddev of the single action components.
        stddev_action_a = actions[:, 0, 0, 0].std()  # batch item 0, action-component (0,0)
        self.assertAlmostEqual(stddev_action_a, 0.0, places=1)
        stddev_action_b = actions[:, 1, 1, 0].std()  # batch item 1, action-component (1,0)
        self.assertAlmostEqual(stddev_action_b, 0.0, places=1)
        stddev_action_c = actions[:, 0, 0, 1].std()  # batch item 0, action-component (0,1)
        self.assertAlmostEqual(stddev_action_c, 0.0, places=1)
        stddev_action_d = actions[:, 1, 1, 1].std()  # batch item 1, action-component (1,1)
        self.assertAlmostEqual(stddev_action_d, 0.0, places=1)
        self.assertAlmostEqual(actions.std(), 1.0, places=0)

    def test_exploration_with_continuous_action_space(self):
        # TODO not portable, redo with more general mean/stddev checks over a sample of distributed outputs.
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

        @rlgraph_api(component=exploration_pipeline)
        def get_action(self_, nn_output):
            _, parameters, _ = action_adapter.get_logits_probabilities_log_probs(nn_output)
            sample_stochastic = distribution.sample_stochastic(parameters)
            sample_deterministic = distribution.sample_deterministic(parameters)
            action = exploration.get_action(sample_stochastic, sample_deterministic)
            return action

        @rlgraph_api(component=exploration_pipeline)
        def get_noise(self_):
            return exploration.noise_component.get_noise()

        test = ComponentTest(component=exploration_pipeline, input_spaces=dict(nn_output=nn_output_space),
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
