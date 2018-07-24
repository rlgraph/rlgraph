# Copyright 2018 The YARL-Project, All Rights Reserved.
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

from yarl.components.helpers.v_trace_function import VTraceFunction
from yarl.spaces import *
from yarl.tests import ComponentTest


class TestVTraceFunctions(unittest.TestCase):

    def test_v_trace_function(self):
        v_trace_function = VTraceFunction()

        batch_x_time_space = FloatBox(add_batch_rank=True, add_time_rank=True)
        batch_space = FloatBox(add_batch_rank=True, add_time_rank=False)
        input_spaces = dict(
            # Log rhos, discounts, rewards, values, bootstrapped_value.
            calc_v_trace_values=[batch_x_time_space, batch_x_time_space, batch_x_time_space, batch_x_time_space,
                                 batch_space]
        )

        test = ComponentTest(component=v_trace_function, input_spaces=input_spaces)

        # Batch of size=2, time-steps=3.
        input_ = [
            # log_rhos
            np.array([[1.0, -0.4, 1.5], [0.65, 0.2, -1.0]]),
            # discounts
            np.array([[0.99, 0.97, 0.5], [0.99, 0.97, 0.5]]),
            # rewards
            np.array([[1.0, 2.0, -5.0], [0.0, 1.0, 2.0]]),
            # values
            np.array([[2.3, 1.563, 0.9], [-1.1, -2.0, -0.3]]),
            # bootstrapped value
            np.array([2.3, -1.0])
        ]

        """
        Calculation:
        """
        expected_vs = np.array([])
        expected_advantages = np.array([])

        test.test(("calc_v_trace_values", input_), expected_outputs=[expected_vs, expected_advantages])
