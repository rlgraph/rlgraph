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

from rlgraph.components import BatchApply, DenseLayer
from rlgraph.spaces import *
from rlgraph.tests import ComponentTest
from rlgraph.utils.numpy import dense_layer


class TestBatchApply(unittest.TestCase):
    """
    Tests the BatchApply Component.
    """
    def test_batch_apply_component_with_simple_input_space(self):
        input_space = FloatBox(shape=(3,), add_batch_rank=True, add_time_rank=True)
        sample = input_space.sample(size=(5, 10))
        sample_folded = np.reshape(sample, newshape=(50, 3))

        sub_component = DenseLayer(units=4, biases_spec=False)

        batch_apply = BatchApply(sub_component=sub_component, api_method_name="apply")
        test = ComponentTest(component=batch_apply, input_spaces=dict(input_=input_space))
        weights = test.read_variable_values(batch_apply.variables["batch-apply/dense-layer/dense/kernel"])

        expected = dense_layer(sample_folded, weights)
        expected = np.reshape(expected, newshape=(5, 10, 4))

        test.test(("apply", sample), expected_outputs=expected)

    def test_batch_apply_component_with_dict_input_space(self):
        return
        input_space = FloatBox(shape=(3,), add_batch_rank=True, add_time_rank=True)
        sample = input_space.sample(size=(5, 10))
        sample_folded = np.reshape(sample, newshape=(50, 3))

        sub_component = DenseLayer(units=4, biases_spec=False)

        batch_apply = BatchApply(sub_component=sub_component, api_method_name="apply")
        test = ComponentTest(component=batch_apply, input_spaces=dict(input_=input_space))
        weights = test.read_variable_values(batch_apply.variables["batch-apply/dense-layer/dense/kernel"])

        expected = dense_layer(sample_folded, weights)
        expected = np.reshape(expected, newshape=(5, 10, 4))

        test.test(("apply", sample), expected_outputs=expected)

