# Copyright 2018/2019 The Rlgraph Authors, All Rights Reserved.
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

from rlgraph.components.common.softmax import Softmax
from rlgraph.spaces import FloatBox, IntBox, Dict
from rlgraph.tests.component_test import ComponentTest
from rlgraph.utils.numpy import softmax as softmax_


class TestSoftmax(unittest.TestCase):

    def test_softmax_on_simple_inputs(self):
        softmax = Softmax()
        input_space = FloatBox(shape=(2, 2, 3), add_batch_rank=True)
        test = ComponentTest(component=softmax, input_spaces=dict(logits=input_space))

        # Batch=5
        inputs = input_space.sample(5)
        expected = softmax_(inputs)
        test.test(("softmax", inputs), expected_outputs=(expected, np.log(expected)))

    def test_softmax_on_complex_inputs(self):
        softmax = Softmax()
        input_space = Dict(dict(a=FloatBox(shape=(4, 5)), b=FloatBox(shape=(3,))),
                                add_batch_rank=True, add_time_rank=True)
        test = ComponentTest(component=softmax, input_spaces=dict(logits=input_space))

        inputs = input_space.sample(size=(4, 5))
        expected = dict(
            a=softmax_(inputs["a"]),
            b=softmax_(inputs["b"])
        )
        expected_logs = dict(
            a=np.log(expected["a"]),
            b=np.log(expected["b"])
        )
        test.test(("softmax", inputs), expected_outputs=(expected, expected_logs), decimals=5)

