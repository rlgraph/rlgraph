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

import unittest
import tensorflow as tf

from rlgraph.components.optimizers import GradientDescentOptimizer
from rlgraph.spaces import Tuple, FloatBox, Dict
from rlgraph.tests import ComponentTest


class TestLocalOptimizers(unittest.TestCase):

    def test_calculate_gradients(self):
        return
        optimizer = GradientDescentOptimizer(learning_rate=0.01)

        x = tf.Variable(2, name='x', dtype=tf.float32)
        log_x = tf.log(x)
        loss = tf.square(x=log_x)

        test = ComponentTest(component=optimizer, input_spaces=dict(
            loss=FloatBox(),
            variables=Dict({"x": FloatBox()}),
            loss_per_item=FloatBox(add_batch_rank=True),
            grads_and_vars=Tuple(Tuple(float, float))
        ))

        print(test.test(("calculate_gradients", [dict(x=x), loss]), expected_outputs=None))

    def test_apply_gradients(self):
        return
        optimizer = GradientDescentOptimizer(learning_rate=0.01)
        x = tf.Variable(2, name='x', dtype=tf.float32)
        log_x = tf.log(x)
        loss = tf.square(x=log_x)

        grads_and_vars = self.optimizer._graph_fn_calculate_gradients(variables=[x], loss=loss)
        step = self.optimizer._graph_fn_apply_gradients(grads_and_vars)
        print(step)