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
import tensorflow as tf

from rlgraph.components.optimizers import GradientDescentOptimizer
from rlgraph.spaces import Tuple, FloatBox, Dict
from rlgraph.tests import ComponentTest, DummyWithOptimizer


class TestLocalOptimizers(unittest.TestCase):

    def test_calculate_gradients(self):
        component = DummyWithOptimizer()

        test = ComponentTest(component=component, input_spaces=dict(
            input_=FloatBox(add_batch_rank=True)
        ))

        print(test.test(("calc_grads", np.ndarray([1.0, 2.0, 3.0, 4.0])), expected_outputs=None))

    def test_apply_gradients(self):
        return
        optimizer = GradientDescentOptimizer(learning_rate=0.01)
        x = tf.Variable(2, name='x', dtype=tf.float32)
        log_x = tf.log(x)
        loss = tf.square(x=log_x)

        grads_and_vars = self.optimizer._graph_fn_calculate_gradients(variables=[x], loss=loss)
        step = self.optimizer._graph_fn_apply_gradients(grads_and_vars)
        print(step)