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

import unittest

import numpy as np
from six.moves import xrange as range_

from rlgraph.components.action_adapters.categorical_distribution_adapter import CategoricalDistributionAdapter
from rlgraph.components.component import Component
from rlgraph.components.distributions import Categorical, Normal
from rlgraph.components.explorations.exploration import Exploration
from rlgraph.spaces import *
from rlgraph.tests import ComponentTest
from rlgraph.utils.decorators import rlgraph_api, graph_fn
from rlgraph.utils.ops import DataOpDict


class TestExplorations(unittest.TestCase):

    def test_exploration_with_discrete_action_space(self):
        nn_output_space = FloatBox(shape=(13,), add_batch_rank=True)
        time_percentage_space = FloatBox(add_batch_rank=True)
        # 2x2 action-pick, each composite action with 5 categories.
        action_space = IntBox(5, shape=(2, 2), add_batch_rank=True)

        # Our distribution to go into the Exploration object.
        distribution = Categorical()
        action_adapter = CategoricalDistributionAdapter(action_space=action_space)

        exploration = Exploration.from_spec(dict(
            epsilon_spec=dict(
                decay_spec=dict(
                    type="linear_decay",
                    from_=1.0,
                    to_=0.0
                )
            )
        ))
        # The Component to test.
        exploration_pipeline = Component(action_adapter, distribution, exploration, scope="exploration-pipeline")

        @rlgraph_api(component=exploration_pipeline)
        def get_action(self_, nn_output, time_percentage):
            out = action_adapter.get_parameters(nn_output)
            sample = distribution.sample_deterministic(out["parameters"])
            action = exploration.get_action(sample, time_percentage)
            return action

        test = ComponentTest(
            component=exploration_pipeline,
            input_spaces=dict(nn_output=nn_output_space, time_percentage=float),
            action_space=action_space
        )

        # With exploration: Check, whether actions are equally distributed.
        nn_outputs = nn_output_space.sample(2)
        time_percentages = time_percentage_space.sample(30)
        # Collect action-batch-of-2 for each of our various random time steps.
        # Each action is an int box of shape=(2,2)
        actions = np.ndarray(shape=(30, 2, 2, 2), dtype=np.int)
        for i, time_percentage in enumerate(time_percentages):
            actions[i] = test.test(("get_action", [nn_outputs, time_percentage]), expected_outputs=None)

        # Assert some distribution of the actions.
        mean_action = actions.mean()
        stddev_action = actions.std()
        self.assertAlmostEqual(mean_action, 2.0, places=0)
        self.assertAlmostEqual(stddev_action, 1.0, places=0)

        # Without exploration (epsilon is force-set to 0.0): Check, whether actions are always the same
        # (given same nn_output all the time).
        nn_outputs = nn_output_space.sample(2)
        time_percentages = time_percentage_space.sample(30) + 0.95
        # Collect action-batch-of-2 for each of our various random time steps.
        # Each action is an int box of shape=(2,2)
        actions = np.ndarray(shape=(30, 2, 2, 2), dtype=np.int)
        for i, time_percentage in enumerate(time_percentages):
            actions[i] = test.test(("get_action", [nn_outputs, time_percentage]), expected_outputs=None)

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

    def test_exploration_with_discrete_container_action_space(self):
        nn_output_space = FloatBox(shape=(12,), add_batch_rank=True)
        time_percentage_space = FloatBox(add_batch_rank=True)
        # Some container action space.
        action_space = Dict(dict(a=IntBox(3), b=IntBox(2), c=IntBox(4)), add_batch_rank=True)

        # Our distribution to go into the Exploration object.
        distribution_a = Categorical(scope="d_a")
        distribution_b = Categorical(scope="d_b")
        distribution_c = Categorical(scope="d_c")
        action_adapter_a = CategoricalDistributionAdapter(action_space=action_space["a"], scope="aa_a")
        action_adapter_b = CategoricalDistributionAdapter(action_space=action_space["b"], scope="aa_b")
        action_adapter_c = CategoricalDistributionAdapter(action_space=action_space["c"], scope="aa_c")

        exploration = Exploration.from_spec(dict(
            epsilon_spec=dict(
                decay_spec=dict(
                    type="linear_decay",
                    from_=1.0,
                    to_=0.0
                )
            )
        ))
        # The Component to test.
        exploration_pipeline = Component(
            action_adapter_a, action_adapter_b, action_adapter_c, distribution_a, distribution_b, distribution_c,
            exploration, scope="exploration-pipeline"
        )

        @rlgraph_api(component=exploration_pipeline)
        def get_action(self_, nn_output, time_percentage):
            out_a = action_adapter_a.get_parameters(nn_output)
            out_b = action_adapter_b.get_parameters(nn_output)
            out_c = action_adapter_c.get_parameters(nn_output)
            sample_a = distribution_a.sample_deterministic(out_a["parameters"])
            sample_b = distribution_b.sample_deterministic(out_b["parameters"])
            sample_c = distribution_c.sample_deterministic(out_c["parameters"])
            sample = self_._graph_fn_merge_actions(sample_a, sample_b, sample_c)
            action = exploration.get_action(sample, time_percentage)
            return action

        @graph_fn(component=exploration_pipeline)
        def _graph_fn_merge_actions(self, a, b, c):
            return DataOpDict(a=a, b=b, c=c)

        test = ComponentTest(
            component=exploration_pipeline,
            input_spaces=dict(nn_output=nn_output_space, time_percentage=float),
            action_space=action_space
        )

        # With (random) exploration (in the beginning of decay interval): Check, whether actions are equally and
        # more or less randomly distributed.
        batch_size = 2
        num_time_steps = 30
        nn_outputs = nn_output_space.sample(batch_size)
        time_percentages = np.maximum(time_percentage_space.sample(num_time_steps) - 0.8, 0)
        # Collect action-batch-of-2 for each of our various random time steps.
        actions_a = np.ndarray(shape=(num_time_steps, batch_size), dtype=np.int)
        actions_b = np.ndarray(shape=(num_time_steps, batch_size), dtype=np.int)
        actions_c = np.ndarray(shape=(num_time_steps, batch_size), dtype=np.int)
        for i, t in enumerate(time_percentages):
            a = test.test(("get_action", [nn_outputs, t]), expected_outputs=None)
            actions_a[i] = a["a"]
            actions_b[i] = a["b"]
            actions_c[i] = a["c"]

        # Assert some distribution of the actions.
        mean_action_a = actions_a.mean()
        stddev_action_a = actions_a.std()
        self.assertAlmostEqual(mean_action_a, 1.0, places=0)
        self.assertAlmostEqual(stddev_action_a, 1.0, places=0)
        mean_action_b = actions_b.mean()
        stddev_action_b = actions_b.std()
        self.assertAlmostEqual(mean_action_b, 0.5, places=0)
        self.assertAlmostEqual(stddev_action_b, 0.5, places=0)
        mean_action_c = actions_c.mean()
        stddev_action_c = actions_c.std()
        self.assertAlmostEqual(mean_action_c, 1.5, places=0)
        self.assertAlmostEqual(stddev_action_c, 1.0, places=0)

        # Without exploration (epsilon is force-set to 0.0): Check, whether actions are always the same
        # (given same nn_output all the time).
        nn_outputs = nn_output_space.sample(batch_size)
        time_percentages = time_percentage_space.sample(num_time_steps) + 1.0
        # Collect action-batch-of-2 for each of our various random time steps.
        actions_a = np.ndarray(shape=(num_time_steps, batch_size), dtype=np.int)
        actions_b = np.ndarray(shape=(num_time_steps, batch_size), dtype=np.int)
        actions_c = np.ndarray(shape=(num_time_steps, batch_size), dtype=np.int)
        for i, t in enumerate(time_percentages):
            a = test.test(("get_action", [nn_outputs, t]), expected_outputs=None)
            actions_a[i] = a["a"]
            actions_b[i] = a["b"]
            actions_c[i] = a["c"]

        # Assert zero stddev of the single action components.
        stddev_action = actions_a[:, 0].std()  # batch item 0, action-component a
        self.assertAlmostEqual(stddev_action, 0.0, places=1)
        stddev_action = actions_a[:, 1].std()  # batch item 1, action-component a
        self.assertAlmostEqual(stddev_action, 0.0, places=1)

        stddev_action = actions_b[:, 0].std()  # batch item 0, action-component b
        self.assertAlmostEqual(stddev_action, 0.0, places=1)
        stddev_action = actions_b[:, 1].std()  # batch item 1, action-component b
        self.assertAlmostEqual(stddev_action, 0.0, places=1)

        stddev_action = actions_c[:, 0].std()  # batch item 0, action-component c
        self.assertAlmostEqual(stddev_action, 0.0, places=1)
        stddev_action = actions_c[:, 1].std()  # batch item 1, action-component c
        self.assertAlmostEqual(stddev_action, 0.0, places=1)

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
            parameters = action_adapter.get_adapter_outputs_and_parameters(nn_output)["parameters"]
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
