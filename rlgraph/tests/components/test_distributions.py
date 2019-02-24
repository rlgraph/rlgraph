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

from rlgraph.components.distributions import *
from rlgraph.spaces import *
from rlgraph.utils.numpy import softmax
from rlgraph.tests import ComponentTest, recursive_assert_almost_equal

import numpy as np


class TestDistributions(unittest.TestCase):

    # TODO also not portable to PyTorch due to batch shapes.

    def test_bernoulli(self):
        # Create 5 bernoulli distributions (or a multiple thereof if we use batch-size > 1).
        param_space = FloatBox(shape=(5,), add_batch_rank=True)

        # The Component to test.
        bernoulli = Bernoulli(switched_off_apis={"log_prob", "kl_divergence"})
        input_spaces = dict(
            parameters=param_space,
            deterministic=bool,
        )
        test = ComponentTest(component=bernoulli, input_spaces=input_spaces)

        # Batch of size=6 and deterministic (True).
        input_ = [input_spaces["parameters"].sample(6), True]
        expected = input_[0] > 0.5
        # Sample n times, expect always max value (max likelihood for deterministic draw).
        for _ in range(10):
            test.test(("draw", input_), expected_outputs=expected)
            test.test(("sample_deterministic", input_[0]), expected_outputs=expected)

        # Batch of size=6 and non-deterministic -> expect roughly the mean.
        input_ = [input_spaces["parameters"].sample(6), False]
        outs = []
        for _ in range(20):
            out = test.test(("draw", input_))
            outs.append(out)
            out = test.test(("sample_stochastic", input_[0]))
            outs.append(out)

        recursive_assert_almost_equal(np.mean(outs), 0.5, decimals=1)

    def test_categorical(self):
        # Create 5 categorical distributions of 3 categories each.
        param_space = FloatBox(shape=(5, 3), add_batch_rank=True)

        # The Component to test.
        categorical = Categorical(switched_off_apis={"log_prob", "kl_divergence"})
        input_spaces = dict(
            parameters=param_space,
            deterministic=bool,
        )
        test = ComponentTest(component=categorical, input_spaces=input_spaces)

        # Batch of size=3 and deterministic (True).
        input_ = [input_spaces["parameters"].sample(3), True]
        expected = np.argmax(input_[0], axis=-1)
        # Sample n times, expect always max value (max likelihood for deterministic draw).
        for _ in range(10):
            test.test(("draw", input_), expected_outputs=expected)
            test.test(("sample_deterministic", input_[0]), expected_outputs=expected)

        # Batch of size=3 and non-deterministic -> expect roughly the mean.
        input_ = [input_spaces["parameters"].sample(3), False]
        outs = []
        for _ in range(20):
            out = test.test(("draw", input_))
            outs.append(out)
            out = test.test(("sample_stochastic", input_[0]))
            outs.append(out)

        recursive_assert_almost_equal(np.mean(outs), 1.0, decimals=1)

    def test_normal(self):
        # Create 5 normal distributions (2 parameters (mean and stddev) each).
        param_space = Tuple(
            FloatBox(shape=(5,)),  # mean
            FloatBox(shape=(5,)),  # stddev
            add_batch_rank=True
        )
        input_spaces = dict(
            parameters=param_space,
            deterministic=bool,
        )

        # The Component to test.
        normal = Normal(switched_off_apis={"log_prob", "kl_divergence"})
        test = ComponentTest(component=normal, input_spaces=input_spaces)

        # Batch of size=2 and deterministic (True).
        input_ = [input_spaces["parameters"].sample(2), True]
        expected = input_[0][0]   # 0 = mean
        # Sample n times, expect always mean value (deterministic draw).
        for _ in range(50):
            test.test(("draw", input_), expected_outputs=expected)
            test.test(("sample_deterministic", tuple([input_[0]])), expected_outputs=expected)

        # Batch of size=1 and non-deterministic -> expect roughly the mean.
        input_ = [input_spaces["parameters"].sample(1), False]
        expected = input_[0][0]  # 0 = mean
        outs = []
        for _ in range(50):
            out = test.test(("draw", input_))
            outs.append(out)
            out = test.test(("sample_stochastic", tuple([input_[0]])))
            outs.append(out)

        recursive_assert_almost_equal(np.mean(outs), expected.mean(), decimals=1)

    def test_multivariate_normal(self):
        # Create batch0=n (batch-rank), batch1=2 (can be used for m mixed Gaussians), num-events=3 (trivariate)
        # distributions (2 parameters (mean and stddev) each).
        num_events = 3  # 3=trivariate Gaussian
        num_mixed_gaussians = 2  # 2x trivariate Gaussians (mixed)
        param_space = Tuple(
            FloatBox(shape=(num_mixed_gaussians, num_events)),  # mean
            FloatBox(shape=(num_mixed_gaussians, num_events)),  # diag (variance)
            add_batch_rank=True
        )
        input_spaces = dict(
            parameters=param_space,
            deterministic=bool,
        )

        # The Component to test.
        multivariate_normal = MultivariateNormal(num_events=num_events, switched_off_apis={"log_prob", "kl_divergence"})
        test = ComponentTest(component=multivariate_normal, input_spaces=input_spaces)

        input_ = [input_spaces["parameters"].sample(4), True]
        expected = input_[0][0]  # 0=mean
        # Sample n times, expect always mean value (deterministic draw).
        for _ in range(50):
            test.test(("draw", input_), expected_outputs=expected)
            test.test(("sample_deterministic", tuple([input_[0]])), expected_outputs=expected)

        # Batch of size=1 and non-deterministic -> expect roughly the mean.
        input_ = [input_spaces["parameters"].sample(1), False]
        expected = input_[0][0]  # 0=mean
        outs = []
        for _ in range(50):
            out = test.test(("draw", input_))
            outs.append(out)
            out = test.test(("sample_stochastic", tuple([input_[0]])))
            outs.append(out)

        recursive_assert_almost_equal(np.mean(outs), expected.mean(), decimals=1)

    def test_beta(self):
        # Create 5 beta distributions (2 parameters (alpha and beta) each).
        param_space = Tuple(
            FloatBox(shape=(5,)),  # alpha
            FloatBox(shape=(5,)),  # beta
            add_batch_rank=True
        )
        input_spaces = dict(
            parameters=param_space,
            deterministic=bool,
        )

        # The Component to test.
        beta_distribution = Beta(switched_off_apis={"log_prob", "kl_divergence"})
        test = ComponentTest(component=beta_distribution, input_spaces=input_spaces)

        # Batch of size=2 and deterministic (True).
        input_ = [input_spaces["parameters"].sample(2), True]
        # Mean for a Beta distribution: 1 / [1 + (beta/alpha)]
        expected = 1.0 / (1.0 + input_[0][1] / input_[0][0])
        # Sample n times, expect always mean value (deterministic draw).
        for _ in range(50):
            test.test(("draw", input_), expected_outputs=expected)
            test.test(("sample_deterministic", tuple([input_[0]])), expected_outputs=expected)

        # Batch of size=1 and non-deterministic -> expect roughly the mean.
        input_ = [input_spaces["parameters"].sample(1), False]
        expected = 1.0 / (1.0 + input_[0][1] / input_[0][0])
        outs = []
        for _ in range(50):
            out = test.test(("draw", input_))
            outs.append(out)
            out = test.test(("sample_stochastic", tuple([input_[0]])))
            outs.append(out)

        recursive_assert_almost_equal(np.mean(outs), expected.mean(), decimals=1)

    def test_mixture(self):
        # Create a mixture distribution consisting of 3 bivariate normals.
        num_distributions = 3
        num_events_per_multivariate = 2  # 2=bivariate
        param_space = Dict(
            {
                "categorical": FloatBox(shape=(num_distributions,)),
                "parameters0": Tuple(
                    FloatBox(shape=(num_events_per_multivariate,)),  # mean
                    FloatBox(shape=(num_events_per_multivariate,)),  # diag
                ),
                "parameters1": Tuple(
                    FloatBox(shape=(num_events_per_multivariate,)),  # mean
                    FloatBox(shape=(num_events_per_multivariate,)),  # diag
                ),
                "parameters2": Tuple(
                    FloatBox(shape=(num_events_per_multivariate,)),  # mean
                    FloatBox(shape=(num_events_per_multivariate,)),  # diag
                ),
            },
            add_batch_rank=True
        )
        input_spaces = dict(
            parameters=param_space,
            deterministic=bool,
        )

        # The Component to test.
        mixture = MixtureDistribution(
            # Try different spec types.
            MultivariateNormal(), "multi-variate-normal", "multivariate_normal",
            switched_off_apis={"entropy", "log_prob", "kl_divergence"}
        )
        test = ComponentTest(component=mixture, input_spaces=input_spaces)

        # Batch of size=n and deterministic (True).
        input_ = [input_spaces["parameters"].sample(1), True]
        # Make probs for categorical.
        input_[0]["categorical"] = softmax(input_[0]["categorical"])

        # Note: Usually, the deterministic draw should return the max-likelihood value
        # Max-likelihood for a 3-Mixed Bivariate: mean-of-argmax(categorical)()
        # argmax = np.argmax(input_[0]["categorical"], axis=-1)
        #expected = np.array([input_[0]["parameters{}".format(idx)][0][i] for i, idx in enumerate(argmax)])
        #    input_[0]["categorical"][:, 1:2] * input_[0]["parameters1"][0] + \
        #    input_[0]["categorical"][:, 2:3] * input_[0]["parameters2"][0]

        # The mean value is a 2D vector (bivariate distribution).
        expected = input_[0]["categorical"][:, 0:1] * input_[0]["parameters0"][0] + \
            input_[0]["categorical"][:, 1:2] * input_[0]["parameters1"][0] + \
            input_[0]["categorical"][:, 2:3] * input_[0]["parameters2"][0]

        for _ in range(50):
            test.test(("draw", input_), expected_outputs=expected)
            test.test(("sample_deterministic", tuple([input_[0]])), expected_outputs=expected)

        # Batch of size=1 and non-deterministic -> expect roughly the mean.
        input_ = [input_spaces["parameters"].sample(1), False]
        # Make probs for categorical.
        input_[0]["categorical"] = softmax(input_[0]["categorical"])

        expected = input_[0]["categorical"][:, 0:1] * input_[0]["parameters0"][0] + \
            input_[0]["categorical"][:, 1:2] * input_[0]["parameters1"][0] + \
            input_[0]["categorical"][:, 2:3] * input_[0]["parameters2"][0]
        outs = []
        for _ in range(50):
            out = test.test(("draw", input_))
            outs.append(out)
            out = test.test(("sample_stochastic", tuple([input_[0]])))
            outs.append(out)

        recursive_assert_almost_equal(np.mean(np.array(outs), axis=0), expected, decimals=1)

    def test_squashed_normal(self):
        param_space = Tuple(
            FloatBox(shape=(1,)),
            FloatBox(shape=(1,)),
            add_batch_rank=True
        )
        values_space = FloatBox(shape=(1,), add_batch_rank=True)
        input_spaces = dict(
            parameters=param_space,
            deterministic=bool,
            values=values_space
        )

        squashed_distribution = SquashedNormal(switched_off_apis={"kl_divergence"}, low=-1.0, high=1.0)
        test = ComponentTest(component=squashed_distribution, input_spaces=input_spaces)

        input_ = [param_space.sample(5), values_space.sample(5)]

        out = test.test(("log_prob", input_), expected_outputs=None)

        print(out)
