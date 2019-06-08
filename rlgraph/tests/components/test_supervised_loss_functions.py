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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import numpy as np
import scipy.stats as sts

from rlgraph.components.loss_functions.categorical_cross_entropy_loss import CategoricalCrossEntropyLoss
from rlgraph.components.loss_functions.container_loss_function import ContainerLossFunction
from rlgraph.components.loss_functions.euclidian_distance_loss import EuclidianDistanceLoss
from rlgraph.components.loss_functions.neg_log_likelihood_loss import NegativeLogLikelihoodLoss
from rlgraph.spaces import *
from rlgraph.spaces.space_utils import get_default_distribution_from_space
from rlgraph.tests import ComponentTest
from rlgraph.utils.numpy import softmax


class TestSupervisedLossFunctions(unittest.TestCase):

    def test_categorical_cross_entropy_loss_wo_time_rank(self):
        #time_steps = 3
        labels_space = IntBox(2, shape=(), add_batch_rank=True)  #, add_time_rank=time_steps)
        parameters_space = labels_space.as_one_hot_float_space()
        loss_per_item_space = FloatBox(shape=(), add_batch_rank=True)
        #sequence_length_space = IntBox(low=1, high=time_steps+1, shape=(), add_batch_rank=True)

        categorical_x_entropy_loss_function = CategoricalCrossEntropyLoss()

        test = ComponentTest(
            component=categorical_x_entropy_loss_function,
            input_spaces=dict(
                labels=labels_space,
                loss_per_item=loss_per_item_space,
                #sequence_length=sequence_length_space,
                parameters=parameters_space
            )
        )

        batch_size = 4
        parameters = parameters_space.sample(batch_size)  #, time_steps)))
        probs = softmax(parameters)
        positive_probs = probs[:, 1]  # parameters[:, :, 1]
        labels = labels_space.sample(batch_size)  #, time_steps))

        # Calculate binary x-entropy manually here: −[ylog(p) + (1-y)log(1-p)]
        # iff label (y) is 0: −log(1−[predicted prob for 1])
        # iff label (y) is 1: −log([predicted prob for 1])
        cross_entropy = np.where(labels == 0, -np.log(1.0 - positive_probs), -np.log(positive_probs))

        #sequence_length = sequence_length_space.sample(batch_size)

        # This code here must be adapted to the exact time-rank reduction schema set within the loss function
        # in case there is a time-rank. For now, test w/o time rank.
        #ces = []
        #for batch_item, sl in enumerate(sequence_length):
        #    weight = 0.5
        #    ce_sum = 0.0
        #    for ce in cross_entropy[batch_item][:sl]:
        #        ce_sum += ce * weight
        #        weight += 0.5 / sequence_length[batch_item]
        #    ces.append(ce_sum / sl)

        expected_loss_per_item = cross_entropy  # np.asarray(ces)
        expected_loss = np.mean(expected_loss_per_item, axis=0, keepdims=False)

        test.test(("loss_per_item", [parameters, labels]),  #, sequence_length]),
                  expected_outputs=expected_loss_per_item, decimals=4)
        test.test(("loss_average", expected_loss_per_item), expected_outputs=expected_loss, decimals=4)
        # Both.
        test.test(("loss", [parameters, labels]),  #, sequence_length]),
                  expected_outputs=[expected_loss, expected_loss_per_item], decimals=4)

    def test_euclidian_distance_loss_function_wo_time_rank(self):
        input_space = FloatBox(shape=(5, 4, 3), add_batch_rank=True)
        loss_per_item_space = FloatBox(shape=(), add_batch_rank=True)

        euclidian_distance_loss_function = EuclidianDistanceLoss()

        test = ComponentTest(
            component=euclidian_distance_loss_function,
            input_spaces=dict(
                parameters=input_space,
                labels=input_space,
                loss_per_item=loss_per_item_space
            )
        )

        parameters = input_space.sample(10)
        labels = input_space.sample(10)

        expected_loss_per_item = np.sqrt(np.sum(np.square(parameters - labels), axis=(-1, -2, -3),
                                                keepdims=False))
        expected_loss = np.mean(expected_loss_per_item, axis=0, keepdims=False)

        test.test(("loss_per_item", [parameters, labels]), expected_outputs=expected_loss_per_item, decimals=4)
        test.test(("loss_average", expected_loss_per_item), expected_outputs=expected_loss, decimals=4)
        # Both.
        test.test(("loss", [parameters, labels]), expected_outputs=[expected_loss, expected_loss_per_item], decimals=4)

    def test_neg_log_likelihood_loss_function_w_simple_space(self):
        shape = (5, 4, 3)
        parameters_space = Tuple(FloatBox(shape=shape), FloatBox(shape=shape), add_batch_rank=True)
        labels_space = FloatBox(shape=shape, add_batch_rank=True)
        loss_per_item_space = FloatBox(add_batch_rank=True)

        loss_function = NegativeLogLikelihoodLoss(distribution_spec=get_default_distribution_from_space(labels_space))

        test = ComponentTest(
            component=loss_function,
            input_spaces=dict(
                parameters=parameters_space,
                labels=labels_space,
                loss_per_item=loss_per_item_space
            )
        )

        parameters = parameters_space.sample(10)
        # Make sure stddev params are not too crazy (just like our adapters do clipping for the raw NN output).
        parameters = (parameters[0], np.clip(parameters[1], 0.1, 1.0))
        labels = labels_space.sample(10)

        expected_loss_per_item = np.sum(-np.log(sts.norm.pdf(labels, parameters[0], parameters[1])), axis=(-1, -2, -3))
        expected_loss = np.mean(expected_loss_per_item, axis=0, keepdims=False)

        test.test(("loss_per_item", [parameters, labels]), expected_outputs=expected_loss_per_item, decimals=4)
        test.test(("loss_average", expected_loss_per_item), expected_outputs=expected_loss, decimals=4)
        # Both.
        test.test(("loss", [parameters, labels]), expected_outputs=[expected_loss, expected_loss_per_item], decimals=4)

    def test_neg_log_likelihood_loss_function_w_container_space(self):
        parameters_space = Dict({
            # Make sure stddev params are not too crazy (just like our adapters do clipping for the raw NN output).
            "a": Tuple(FloatBox(shape=(2, 3)), FloatBox(0.5, 1.0, shape=(2, 3))),  # normal (0.0 to 1.0)
            "b": FloatBox(shape=(4,), low=-1.0, high=1.0)  # 4-discrete
        }, add_batch_rank=True)
        labels_space = Dict({
            "a": FloatBox(shape=(2, 3)),
            "b": IntBox(4)
        }, add_batch_rank=True)
        loss_per_item_space = FloatBox(add_batch_rank=True)

        loss_function = NegativeLogLikelihoodLoss(distribution_spec=get_default_distribution_from_space(labels_space))

        test = ComponentTest(
            component=loss_function,
            input_spaces=dict(
                parameters=parameters_space,
                labels=labels_space,
                loss_per_item=loss_per_item_space
            )
        )

        parameters = parameters_space.sample(2)
        # Softmax the discrete params.
        probs_b = softmax(parameters["b"])
        #probs_b = parameters["b"]
        labels = labels_space.sample(2)

        # Expected loss: Sum of all -log(llh)
        log_prob_per_item_a = np.sum(np.log(sts.norm.pdf(labels["a"], parameters["a"][0], parameters["a"][1])), axis=(-1, -2))
        log_prob_per_item_b = np.array([np.log(probs_b[0][labels["b"][0]]), np.log(probs_b[1][labels["b"][1]])])

        expected_loss_per_item = - (log_prob_per_item_a + log_prob_per_item_b)
        expected_loss = np.mean(expected_loss_per_item, axis=0, keepdims=False)

        test.test(("loss_per_item", [parameters, labels]), expected_outputs=expected_loss_per_item, decimals=4)
        test.test(("loss_average", expected_loss_per_item), expected_outputs=expected_loss, decimals=4)
        # Both.
        test.test(("loss", [parameters, labels]), expected_outputs=[expected_loss, expected_loss_per_item], decimals=4)

    def test_container_loss_function(self):
        # Regular float layer output.
        input_space_euclidian = FloatBox(shape=(5, 3))
        # Tuple output (parameters) for a Normal distribution.
        input_space_neg_log_llh = Tuple(FloatBox(shape=(5, 3)), FloatBox(shape=(5, 3)))
        # The predictions are a=output, b=parameters.
        input_space_parameters = Dict({"a": input_space_euclidian, "b": input_space_neg_log_llh}, add_batch_rank=True)

        input_space_labels = Dict(
            {
                "a": input_space_euclidian,
                "b": input_space_neg_log_llh[0]
            }, add_batch_rank=True
        )
        loss_per_item_space = FloatBox(shape=(), add_batch_rank=True)

        container_loss_function = ContainerLossFunction(
            loss_functions_spec=dict(
                a=dict(type="euclidian-distance-loss"),
                b=dict(type="neg-log-likelihood-loss", distribution_spec=dict(type="normal-distribution"))
            ), weights=dict(a=0.2, b=0.4))

        test = ComponentTest(
            component=container_loss_function,
            input_spaces=dict(
                parameters=input_space_parameters,
                labels=input_space_labels,
                loss_per_item=loss_per_item_space
            )
        )

        predictions = input_space_parameters.sample(3)
        labels = input_space_labels.sample(3)

        expected_euclidian = 0.2 * np.sqrt(
            np.sum(np.square(predictions["a"] - labels["a"]), axis=(-1, -2), keepdims=False)
        )
        expected_neg_log_llh = 0.4 * np.sum(
            (- np.log(sts.norm.pdf(labels["b"], predictions["b"][0], predictions["b"][1]))),
            axis=(-1, -2), keepdims=False
        )

        expected_loss_per_item = expected_euclidian + expected_neg_log_llh
        expected_loss = np.mean(expected_loss_per_item, axis=0, keepdims=False)

        test.test(("loss_per_item", [predictions, labels]), expected_outputs=expected_loss_per_item, decimals=4)
        test.test(("loss_average", expected_loss_per_item), expected_outputs=expected_loss, decimals=4)
        # Both.
        test.test(("loss", [predictions, labels]), expected_outputs=[expected_loss, expected_loss_per_item], decimals=4)

