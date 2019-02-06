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

from rlgraph.spaces import FloatBox
from rlgraph.tests import ComponentTest, DummyWithOptimizer, recursive_assert_almost_equal


class TestLocalOptimizers(unittest.TestCase):

    def test_calculate_gradients(self):
        component = DummyWithOptimizer()

        test = ComponentTest(component=component, input_spaces=dict(
            input_=FloatBox(add_batch_rank=True)
        ))

        expected_outputs = [0.73240823, 3.0]
        test.test(("calc_grads"), expected_outputs=expected_outputs)

    def test_apply_gradients(self):
        component = DummyWithOptimizer(variable_value=2.0)

        test = ComponentTest(component=component, input_spaces=dict(
            input_=FloatBox(add_batch_rank=True)
        ))

        expected_grad = 0.69314718
        expected_outputs = [expected_grad, 2.0]
        test.test(("calc_grads"), expected_outputs=expected_outputs)

        # Now apply the grad and check the variable value.
        expected_loss = np.square(np.log(2.0))
        expected_outputs = [None, expected_loss, expected_loss]
        var_values_before = test.read_variable_values(component.variable_registry)
        test.test(("step"), expected_outputs=expected_outputs)

        # Check against variable now. Should change by -learning_rate*grad.
        var_values_after = test.read_variable_values(component.variable_registry)
        expected_new_value = var_values_before["dummy-with-optimizer/variable"] - (
            component.learning_rate * expected_grad
        )
        recursive_assert_almost_equal(
            var_values_after["dummy-with-optimizer/variable"], expected_new_value, decimals=5
        )

