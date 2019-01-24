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

import numpy as np
import unittest

from rlgraph.components.helpers.v_trace_function import VTraceFunction
from rlgraph.spaces import *
from rlgraph.tests import ComponentTest
from rlgraph.utils.numpy import one_hot, softmax


class TestVTraceFunctions(unittest.TestCase):

    time_x_batch_x_2_space = FloatBox(shape=(2,), add_batch_rank=True, add_time_rank=True, time_major=True)
    time_x_batch_x_9_space = FloatBox(shape=(9,), add_batch_rank=True, add_time_rank=True, time_major=True)
    time_x_batch_x_1_space = FloatBox(shape=(1,), add_batch_rank=True, add_time_rank=True, time_major=True)

    def test_v_trace_function(self):
        v_trace_function = VTraceFunction()
        v_trace_function_reference = VTraceFunction(backend="python")

        action_space = IntBox(2, add_batch_rank=True, add_time_rank=True, time_major=True)
        action_space_flat = FloatBox(shape=(2,), add_batch_rank=True, add_time_rank=True, time_major=True)
        input_spaces = dict(
            logits_actions_pi=self.time_x_batch_x_2_space,
            log_probs_actions_mu=self.time_x_batch_x_2_space,
            actions=action_space,
            actions_flat=action_space_flat,
            discounts=self.time_x_batch_x_1_space,
            rewards=self.time_x_batch_x_1_space,
            values=self.time_x_batch_x_1_space,
            bootstrapped_values=self.time_x_batch_x_1_space
        )

        test = ComponentTest(component=v_trace_function, input_spaces=input_spaces)

        size = (3, 2)
        logits_actions_pi = self.time_x_batch_x_2_space.sample(size=size)
        logits_actions_mu = self.time_x_batch_x_2_space.sample(size=size)
        log_probs_actions_mu = np.log(softmax(logits_actions_mu))
        actions = action_space.sample(size=size)
        actions_flat = one_hot(actions, depth=action_space.num_categories)
        # Set some discounts to 0.0 (these will mark the end of episodes, where the value is 0.0).
        discounts = np.random.choice([0.0, 0.99], size=size + (1,), p=[0.2, 0.8])
        rewards = self.time_x_batch_x_1_space.sample(size=size)
        values = self.time_x_batch_x_1_space.sample(size=size)
        bootstrapped_values = self.time_x_batch_x_1_space.sample(size=(1, size[1]))

        input_ = [
            logits_actions_pi, log_probs_actions_mu, actions, actions_flat, discounts, rewards, values,
            bootstrapped_values
        ]

        vs_expected, pg_advantages_expected = v_trace_function_reference._graph_fn_calc_v_trace_values(*input_)

        test.test(("calc_v_trace_values", input_), expected_outputs=[vs_expected, pg_advantages_expected], decimals=4)

    def test_v_trace_function_more_complex(self):
        v_trace_function = VTraceFunction()
        v_trace_function_reference = VTraceFunction(backend="python")

        action_space = IntBox(9, add_batch_rank=True, add_time_rank=True, time_major=True)
        action_space_flat = FloatBox(shape=(9,), add_batch_rank=True, add_time_rank=True, time_major=True)
        input_spaces = dict(
            logits_actions_pi=self.time_x_batch_x_9_space,
            log_probs_actions_mu=self.time_x_batch_x_9_space,
            actions=action_space,
            actions_flat=action_space_flat,
            discounts=self.time_x_batch_x_1_space,
            rewards=self.time_x_batch_x_1_space,
            values=self.time_x_batch_x_1_space,
            bootstrapped_values=self.time_x_batch_x_1_space
        )

        test = ComponentTest(component=v_trace_function, input_spaces=input_spaces)

        size = (100, 16)
        logits_actions_pi = self.time_x_batch_x_9_space.sample(size=size)
        logits_actions_mu = self.time_x_batch_x_9_space.sample(size=size)
        log_probs_actions_mu = np.log(softmax(logits_actions_mu))
        actions = action_space.sample(size=size)
        actions_flat = one_hot(actions, depth=action_space.num_categories)
        # Set some discounts to 0.0 (these will mark the end of episodes, where the value is 0.0).
        discounts = np.random.choice([0.0, 0.99], size=size + (1,), p=[0.1, 0.9])
        rewards = self.time_x_batch_x_1_space.sample(size=size)
        values = self.time_x_batch_x_1_space.sample(size=size)
        bootstrapped_values = self.time_x_batch_x_1_space.sample(size=(1, size[1]))

        input_ = [
            logits_actions_pi, log_probs_actions_mu, actions, actions_flat, discounts, rewards, values,
            bootstrapped_values
        ]

        vs_expected, pg_advantages_expected = v_trace_function_reference._graph_fn_calc_v_trace_values(*input_)

        test.test(("calc_v_trace_values", input_), expected_outputs=[vs_expected, pg_advantages_expected], decimals=4)

