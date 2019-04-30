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
from scipy.stats import norm, beta

from rlgraph.components.distributions import *
from rlgraph.spaces import *
from rlgraph.tests import ComponentTest, recursive_assert_almost_equal
from rlgraph.utils.numpy import softmax


class TestDistributions(unittest.TestCase):

    # TODO also not portable to PyTorch due to batch shapes.

    def test_bernoulli(self):
        # Create 5 bernoulli distributions (or a multiple thereof if we use batch-size > 1).
        param_space = FloatBox(shape=(5,), add_batch_rank=True)
        values_space = BoolBox(shape=(5,), add_batch_rank=True)

        # The Component to test.
        bernoulli = Bernoulli(switched_off_apis={"kl_divergence"})
        input_spaces = dict(
            parameters=param_space,
            values=values_space,
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

        # Test log-likelihood outputs.
        test.test(("log_prob", [
            np.array([[0.1, 0.2, 0.3, 0.4, 0.5]]),
            np.array([[True, False, False, True, True]])
        ]), expected_outputs=np.log(np.array([[0.1, 0.8, 0.7, 0.4, 0.5]])))  # probs that's the result is True

    def test_categorical(self):
        # Create 5 categorical distributions of 3 categories each.
        param_space = FloatBox(shape=(5, 3), add_batch_rank=True)
        values_space = IntBox(3, shape=(5,), add_batch_rank=True)

        # The Component to test.
        categorical = Categorical(switched_off_apis={"kl_divergence"})
        input_spaces = dict(
            parameters=param_space,
            values=values_space,
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

        # Test log-likelihood outputs.
        test.test(("log_prob", [
            np.array([[[0.1, 0.8, 0.1], [0.2, 0.2, 0.6], [0.3, 0.3, 0.4], [0.4, 0.1, 0.5], [0.5, 0.05, 0.45]]]),
            np.array([[1, 2, 0, 1, 1]])
        ]), expected_outputs=np.log(np.array([[0.8, 0.6, 0.3, 0.1, 0.05]])))

    def test_normal(self):
        # Create 5 normal distributions (2 parameters (mean and stddev) each).
        param_space = Tuple(
            FloatBox(shape=(5,)),  # mean
            FloatBox(shape=(5,)),  # stddev
            add_batch_rank=True
        )
        values_space = FloatBox(shape=(5,), add_batch_rank=True)
        input_spaces = dict(
            parameters=param_space,
            values=values_space,
            deterministic=bool,
        )

        # The Component to test.
        normal = Normal(switched_off_apis={"kl_divergence"})
        test = ComponentTest(component=normal, input_spaces=input_spaces)

        # Batch of size=2 and deterministic (True).
        input_ = [param_space.sample(2), True]
        expected = input_[0][0]   # 0 = mean
        # Sample n times, expect always mean value (deterministic draw).
        for _ in range(50):
            test.test(("draw", input_), expected_outputs=expected)
            test.test(("sample_deterministic", tuple([input_[0]])), expected_outputs=expected)

        # Batch of size=1 and non-deterministic -> expect roughly the mean.
        input_ = [param_space.sample(1), False]
        expected = input_[0][0]  # 0 = mean
        outs = []
        for _ in range(50):
            out = test.test(("draw", input_))
            outs.append(out)
            out = test.test(("sample_stochastic", tuple([input_[0]])))
            outs.append(out)

        recursive_assert_almost_equal(np.mean(outs), expected.mean(), decimals=1)

        # Test log-likelihood outputs.
        means = np.array([[0.1, 0.2, 0.3, 0.4, 100.0]])
        stds = np.array([[0.8, 0.2, 0.3, 2.0, 50.0]])
        values = np.array([[1.0, 2.0, 0.4, 10.0, 5.4]])
        test.test(
            ("log_prob", [tuple([means, stds]), values]),
            expected_outputs=np.log(norm.pdf(values, means, stds)), decimals=4
        )

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
        values_space = FloatBox(shape=(num_mixed_gaussians, num_events), add_batch_rank=True)
        input_spaces = dict(
            parameters=param_space,
            values=values_space,
            deterministic=bool,
        )

        # The Component to test.
        multivariate_normal = MultivariateNormal(num_events=num_events, switched_off_apis={"kl_divergence"})
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

        # Test log-likelihood outputs (against scipy).
        means = values_space.sample(2)
        stds = values_space.sample(2)
        values = values_space.sample(2)
        test.test(
            ("log_prob", [tuple([means, stds]), values]),
            # Sum up the individual log-probs as we have a diag (independent) covariance matrix.
            expected_outputs=np.sum(np.log(norm.pdf(values, means, stds)), axis=-1), decimals=4
        )

    def test_beta(self):
        # Create 5 beta distributions (2 parameters (alpha and beta) each).
        param_space = Tuple(
            FloatBox(shape=(5,)),  # alpha
            FloatBox(shape=(5,)),  # beta
            add_batch_rank=True
        )
        values_space = FloatBox(shape=(5,), add_batch_rank=True)
        input_spaces = dict(
            parameters=param_space,
            values=values_space,
            deterministic=bool,
        )

        # The Component to test.
        beta_distribution = Beta(switched_off_apis={"kl_divergence"})
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

        # Test log-likelihood outputs (against scipy).
        means = values_space.sample(1)
        stds = values_space.sample(1)
        values = values_space.sample(1)
        test.test(
            ("log_prob", [tuple([means, stds]), values]),
            expected_outputs=np.log(beta.pdf(values, means, stds)), decimals=4
        )

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
        values_space = FloatBox(shape=(num_events_per_multivariate,), add_batch_rank=True)
        input_spaces = dict(
            parameters=param_space,
            values=values_space,
            deterministic=bool,
        )

        # The Component to test.
        mixture = MixtureDistribution(
            # Try different spec types.
            MultivariateNormal(), "multi-variate-normal", "multivariate_normal",
            switched_off_apis={"entropy", "kl_divergence"}
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

        # Test log-likelihood outputs (against scipy).
        params = param_space.sample(1)
        # Make sure categorical params are softmaxed.
        category_probs = softmax(params["categorical"][0])
        values = values_space.sample(1)
        expected = \
            category_probs[0] * \
            np.sum(np.log(norm.pdf(values[0], params["parameters0"][0][0], params["parameters0"][1][0])), axis=-1) + \
            category_probs[1] * \
            np.sum(np.log(norm.pdf(values[0], params["parameters1"][0][0], params["parameters1"][1][0])), axis=-1) + \
            category_probs[2] * \
            np.sum(np.log(norm.pdf(values[0], params["parameters2"][0][0], params["parameters2"][1][0])), axis=-1)
        test.test(("log_prob", [params, values]), expected_outputs=np.array([expected]), decimals=1)

    def test_squashed_normal(self):
        param_space = Tuple(
            FloatBox(shape=(5,)),
            FloatBox(shape=(5,)),
            add_batch_rank=True
        )
        values_space = FloatBox(shape=(5,), add_batch_rank=True)
        input_spaces = dict(
            parameters=param_space,
            deterministic=bool,
            values=values_space
        )

        low, high = -1.0, 1.0
        squashed_distribution = SquashedNormal(switched_off_apis={"kl_divergence"}, low=low, high=high)
        test = ComponentTest(component=squashed_distribution, input_spaces=input_spaces)

        # Batch of size=2 and deterministic (True).
        input_ = [param_space.sample(2), True]
        expected = ((np.tanh(input_[0][0]) + 1.0) / 2.0) * (high - low) + low   # 0 = mean
        # Sample n times, expect always mean value (deterministic draw).
        for _ in range(50):
            test.test(("draw", input_), expected_outputs=expected)
            test.test(("sample_deterministic", tuple([input_[0]])), expected_outputs=expected)

        # Batch of size=1 and non-deterministic -> expect roughly the mean.
        input_ = [param_space.sample(1), False]
        expected = ((np.tanh(input_[0][0]) + 1.0) / 2.0) * (high - low) + low  # 0 = mean
        outs = []
        for _ in range(50):
            out = test.test(("draw", input_))
            outs.append(out)
            out = test.test(("sample_stochastic", tuple([input_[0]])))
            outs.append(out)

        recursive_assert_almost_equal(np.mean(outs), expected.mean(), decimals=1)

        # Test log-likelihood outputs.
        means = np.array([[0.1, 0.2, 0.3, 0.4, 100.0]])
        stds = np.array([[0.8, 0.2, 0.3, 2.0, 50.0]])
        # Make sure values are within low and high.
        values = np.array([[0.99, 0.2, 0.4, -0.1, -0.8542]])

        # TODO: understand and comment the following formula to get the log-prob.
        # Unsquash values, then get log-llh from regular gaussian.
        unsquashed_values = np.arctanh((values - low) / (high - low) * 2.0 - 1.0)
        log_prob_unsquashed = np.log(norm.pdf(unsquashed_values, means, stds))
        log_prob = log_prob_unsquashed - np.sum(np.log(1 - np.tanh(unsquashed_values) ** 2), axis=-1, keepdims=True)

        test.test(("log_prob", [tuple([means, stds]), values]), expected_outputs=log_prob, decimals=4)

    def test_joint_cumulative_distribution(self):
        param_space = Dict({
            "a": FloatBox(shape=(4,)),  # 4-discrete
            "b": Tuple([FloatBox(shape=(9,)), FloatBox(shape=(9,))])  # 9-variate normal
        }, add_batch_rank=True)

        values_space = Dict({"a": IntBox(4), "b": FloatBox(shape=(9,))}, add_batch_rank=True)

        input_spaces = dict(
            parameters=param_space,
            deterministic=bool,
            values=values_space
        )

        joined_cumulative_distribution = JointCumulativeDistribution(sub_distributions_spec=dict(
            a=Categorical(), b=Normal()
        ), switched_off_apis={"kl_divergence"})
        test = ComponentTest(component=joined_cumulative_distribution, input_spaces=input_spaces)

        input_ = [param_space.sample(5), values_space.sample(5)]

        out = test.test(("log_prob", input_), expected_outputs=None)

        print(out)
