# Copyright 2018/2019 ducandu GmbH. All Rights Reserved.
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

from __future__ import absolute_import, division, print_function

import unittest

import numpy as np

from rlgraph.components.models.supervised_model import SupervisedModel
from rlgraph.spaces import FloatBox
from rlgraph.tests.component_test import ComponentTest


class TestModels(unittest.TestCase):
    """
    Tests for the different Model classes.
    """
    def test_simple_supervised_model(self):
        input_space = FloatBox(shape=(2,), add_batch_rank=True)
        output_space = FloatBox(shape=(1,), add_batch_rank=True)

        # Model needs to "predict" the sum of both input values.
        model = SupervisedModel(
            supervised_predictor_spec=dict(
                network_spec=[{"type": "dense", "units": 3}],
                output_space=output_space
            ),
            loss_function_spec=dict(
                type="euclidian-distance-loss"
            ),
            optimizer_spec=dict(
                type="adam",
                learning_rate=0.005
            )
        )

        test = ComponentTest(component=model, input_spaces=dict(
            nn_inputs=input_space,
            labels=output_space
        ))

        # Test learning capabilities of the model.
        batch_size = 256
        losses = []
        for i in range(100):
            inputs = input_space.sample(batch_size)
            # Labels are always the sum of the inputs.
            labels = np.sum(inputs, axis=-1, keepdims=True)

            out = test.test(("update", [inputs, labels]))
            # Print out and store loss.
            losses.append(out["loss"])
            #print("Epoch {}: Loss={}".format(i, out["loss"]))

        # Make sure we have learnt something.
        self.assertTrue(np.mean(losses[-10:]) < np.mean(losses[:10]))

        test.terminate()
