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

import tensorflow as tf

from functools import partial

from yarl.utils.util import dtype
from yarl.components import Component
from yarl.spaces import ContainerSpace


class Distribution(Component):
    """
    A distribution wrapper class that can incorporate a backend-specific distribution object that gets its parameters
    from an external source (e.g. a NN).

    API:
    ins:
        parameters (numeric): The parameters of the distribution (e.g. mean and variance for a Gaussian).
            The Space of parameters must have a batch-rank.
        max_likelihood (bool): Whether to sample or to get the max-likelihood value (deterministic).
    outs:
        draw (numeric): Draws a sample from the distribution.
        entropy (float): The entropy value of the distribution.
    """
    def __init__(self, scope="distribution", **kwargs):
        super(Distribution, self).__init__(scope=scope, **kwargs)

        # Define a generic Distribution interface.
        self.define_inputs("parameters", "max_likelihood")
        self.define_outputs("draw", "entropy")
        # "distribution" will be an internal Socket used to connect the Computations with each other.
        self.add_computation("parameters", "distribution", self._computation_parameterize)
        self.add_computation(["distribution", "max_likelihood"], "draw", self._computation_draw)
        self.add_computation("distribution", "entropy", self._computation_entropy)

    def create_variables(self, input_spaces):
        in_space = input_spaces["parameters"]
        # a) Must not be ContainerSpace (not supported yet for Distributions, doesn't seem to make sense).
        assert not isinstance(in_space, ContainerSpace), "ERROR: Cannot handle container input Spaces " \
                                                         "in distribution '{}' (atm; may soon do)!".format(self.name)
        # b) All input Spaces need batch ranks.
        assert in_space.has_batch_rank,\
            "ERROR: Space in Socket 'input' to layer '{}' must have a batch rank (0th position)!".format(self.name)

    def _computation_parameterize(self, parameters):
        """
        Parameterizes this distribution (normally from an NN-output vector). Returns the backend-distribution object
        (a DataOp).

        Args:
            parameters (DataOp): The input used to parameterize this distribution. This is normally a cleaned up
                NN-output that, for example, can hold the two values for mean and variance for a univariate Gaussian
                distribution.

        Returns:
            DataOp: The parameterized backend-specific distribution object.
        """
        raise NotImplementedError

    def _computation_draw(self, distribution, max_likelihood):
        """
        Takes a sample from the (already parameterized) distribution. The parameterization also includes a possible
        batch size.

        Args:
            distribution (DataOp): The (already parameterized) backend-specific distribution DataOp to use for
                sampling. This is simply the output of `self._computation_parameterize`.
            max_likelihood (bool): Whether to return the maximum-likelihood result, instead of a random sample.
                Can be used to pick deterministic actions from discrete ("greedy") or continuous (mean-value)
                distributions.

        Returns:
            DataOp: The taken sample(s).
        """
        return tf.where(condition=max_likelihood,
                        x=partial(self._max_likelihood, distribution),
                        y=partial(self._sampled, distribution))

    def _max_likelihood(self, distribution):
        """
        Returns the maximum-likelihood value for a given distribution.

        Args:
            distribution (DataOp): The (already parameterized) backend-specific distribution whose max-likelihood value
                to calculate. This is simply the output of `self._computation_parameterize`.

        Returns:
            DataOp: The max-likelihood value.
        """
        raise NotImplementedError

    @staticmethod
    def _sampled(distribution):
        """
        Returns an actual sample for a given distribution.

        Args:
            distribution (DataOp): The (already parameterized) backend-specific distribution from which a sample
                should be drawn. This is simply the output of `self._computation_parameterize`.

        Returns:
            DataOp: The drawn sample.
        """
        return distribution.sample()

    @staticmethod
    def _computation_entropy(distribution):
        """
        Returns the DataOp holding the entropy value of the distribution.

        Args:
            distribution (DataOp): The (already parameterized) backend-specific distribution whose entropy to
                calculate. This is simply the output of `self._computation_parameterize`.

        Returns:
            DataOp: The distribution's entropy.
        """
        return distribution.entropy()


class Bernoulli(Distribution):
    """
    A Bernoulli distribution object defined by a single value p, the probability for True (rather than False).
    """
    def __init__(self, scope="bernoulli", **kwargs):
        super(Bernoulli, self).__init__(scope=scope, **kwargs)

    def _computation_parameterize(self, prob):
        """
        Args:
            prob (DataOp): The p value (probability that distribution returns True).
        """
        # TODO: Move this into some sort of NN-output-cleanup component. This is repetitive stuff.
        ## Clamp raw_input between 0 and 1 to make it interpretable as a probability.
        #p = tf.sigmoid(x=flat_input)
        ## Clamp to avoid 0.0 or 1.0 probabilities (adds numerical stability).
        #p = tf.clip_by_value(p, clip_value_min=SMALL_NUMBER, clip_value_max=(1.0 - SMALL_NUMBER))

        return tf.distributions.Bernoulli(probs=prob, dtype=dtype("bool"))

    def _max_likelihood(self, distribution):
        return distribution.prob >= 0.5


class Categorical(Distribution):
    """
    A categorical distribution object defined by a n values {p0, p1, ...} that add up to 1, the probabilities
    for picking one of the n categories.
    """
    def __init__(self, scope="categorical", **kwargs):
        super(Categorical, self).__init__(scope=scope, **kwargs)

    def _computation_parameterize(self, probs):
        return tf.distributions.Categorical(probs=probs, dtype=dtype("int"))

    def _max_likelihood(self, distribution):
        return tf.argmax(input=distribution.probs, axis=-1, output_type=dtype("int"))

