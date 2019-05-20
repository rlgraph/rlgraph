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

from rlgraph.components.common.time_dependent_parameters import TimeDependentParameter
from rlgraph.spaces import *
from rlgraph.tests import ComponentTest


class TestParameters(unittest.TestCase):
    """
    Tests time-step dependent TimeDependentParameter Component classes.
    """

    input_space_pct = dict(time_percentage=FloatBox(add_batch_rank=True))

    def test_constant_parameter(self):
        constant = TimeDependentParameter.from_spec(2.0)
        test = ComponentTest(component=constant, input_spaces=self.input_space_pct)

        input_ = np.array([0.5, 0.1, 1.0, 0.9, 0.02, 0.01, 0.99, 0.23])

        test.test(("get", input_), expected_outputs=[
            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0
        ])

    def test_linear_parameter(self):
        linear_parameter = TimeDependentParameter.from_spec((2.0, 0.5))
        test = ComponentTest(component=linear_parameter, input_spaces=self.input_space_pct)

        input_ = np.array([0.5, 0.1, 1.0, 0.9, 0.02, 0.01, 0.99, 0.23])

        test.test(("get", input_), expected_outputs=2.0 - input_ * (2.0 - 0.5))

    def test_linear_parameter_using_global_timestep(self):
        linear_parameter = TimeDependentParameter.from_spec("linear-decay", from_=2.0, to_=0.5, max_time_steps=100)
        test = ComponentTest(component=linear_parameter, input_spaces=None)

        # Call without any parameters -> force component to use GLOBAL_STEP, which should be 0 right now -> no decay.
        for _ in range(10):
            test.test("get", expected_outputs=2.0)

    def test_polynomial_parameter(self):
        polynomial_parameter = TimeDependentParameter.from_spec(type="polynomial-decay", from_=2.0, to_=0.5, power=2.0)
        test = ComponentTest(component=polynomial_parameter, input_spaces=self.input_space_pct)

        input_ = np.array([0.5, 0.1, 1.0, 0.9, 0.02, 0.01, 0.99, 0.23])

        test.test(("get", input_), expected_outputs=(2.0 - 0.5) * (1.0 - input_) ** 2 + 0.5)

    def test_polynomial_parameter_using_global_timestep(self):
        polynomial_parameter = TimeDependentParameter.from_spec("polynomial-decay", from_=3.0, to_=0.5, max_time_steps=100)
        test = ComponentTest(component=polynomial_parameter, input_spaces=None)

        # Call without any parameters -> force component to use GLOBAL_STEP, which should be 0 right now -> no decay.
        for _ in range(10):
            test.test("get", expected_outputs=3.0)

    def test_exponential_parameter(self):
        exponential_parameter = TimeDependentParameter.from_spec(type="exponential-decay", from_=2.0, to_=0.5, decay_rate=0.5)
        test = ComponentTest(component=exponential_parameter, input_spaces=self.input_space_pct)

        input_ = np.array([0.5, 0.1, 1.0, 0.9, 0.02, 0.01, 0.99, 0.23])

        test.test(("get", input_), expected_outputs=0.5 + (2.0 - 0.5) * 0.5 ** input_)

    def test_exponential_parameter_using_global_timestep(self):
        exponential_parameter = TimeDependentParameter.from_spec("exponential-decay", from_=3.0, to_=0.5, max_time_steps=100)
        test = ComponentTest(component=exponential_parameter, input_spaces=None)

        # Call without any parameters -> force component to use GLOBAL_STEP, which should be 0 right now -> no decay.
        for _ in range(10):
            test.test("get", expected_outputs=3.0)
