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

import unittest

import tensorflow as tf

from yarl.components.distributions import Bernoulli, Categorical
from yarl.spaces import *
from yarl.tests import ComponentTest

import numpy as np


class TestDistributions(unittest.TestCase):

    def test_bernoulli(self):
        # Create 5 bernoulli distributions (or a multiple thereof if we use batch-size > 1).
        param_space = FloatBox(shape=(5,), add_batch_rank=True)
        max_likelihood_space = BoolBox()

        # The Component to test.
        bernoulli = Bernoulli()
        test = ComponentTest(component=bernoulli, input_spaces=dict(parameters=param_space,
                                                                    max_likelihood=max_likelihood_space))

        # Batch of size=1 and deterministic.
        input_ = {
            "parameters": np.array([[0.5, 0.99, 0.0, 0.2, 0.3]]),
            "max_likelihood": True
        }
        expected = np.array([[True, True, False, False, False]])
        test.test(out_socket_name="draw", inputs=input_, expected_outputs=expected)

        # Batch of size=2 and non-deterministic -> expect always the same result when we seed tf (done automatically
        # by the ComponentTest object).
        input_ = {
            "parameters": np.array([[0.1, 0.3, 0.6, 0.71, 0.001], [0.9, 0.998, 0.9999, 0.0001, 0.345678]]),
            "max_likelihood": False
        }
        expected = np.array([[True, True, False, True, False], [True, True, True, False, True]])
        test.test(out_socket_name="draw", inputs=input_, expected_outputs=expected)

    def test_categorical(self):
        # Create 5 categorical distributions of 3 categories each.
        param_space = FloatBox(shape=(5, 3), add_batch_rank=True)
        max_likelihood_space = BoolBox()

        # The Component to test.
        categorical = Categorical()
        test = ComponentTest(component=categorical, input_spaces=dict(parameters=param_space,
                                                                      max_likelihood=max_likelihood_space))

        # Batch of size=1 and deterministic.
        input_ = {
            "parameters": np.array([[[0.5, 0.25, 0.25],
                                     [0.98, 0.01, 0.01],
                                     [0.0, 0.6, 0.4],
                                     [0.2, 0.25, 0.55],
                                     [0.3, 0.3, 0.4]
                                     ]]),
            "max_likelihood": True
        }
        expected = np.array([[0, 0, 1, 2, 2]])
        test.test(out_socket_name="draw", inputs=input_, expected_outputs=expected)

        # Batch of size=2 and non-deterministic -> expect always the same result when we seed tf (done automatically
        # by the ComponentTest object).
        input_ = {
            "parameters": np.array([[[0.3, 0.25, 0.45],
                                     [0.96, 0.02, 0.02],
                                     [0.0, 0.5, 0.5],
                                     [0.1, 0.85, 0.05],
                                     [0.6, 0.1, 0.3]
                                     ],
                                    [[0.65, 0.05, 0.3],
                                     [0.0001, 0.0001, 0.9998],
                                     [0.82, 0.12, 0.06],
                                     [0.5, 0.0001, 0.4999],
                                     [0.333, 0.333, 0.334]
                                     ]
                                    ]),
            "max_likelihood": False
        }
        expected = np.array([[0, 0, 1, 1, 0], [0, 2, 1, 2, 1]])
        test.test(out_socket_name="draw", inputs=input_, expected_outputs=expected)

