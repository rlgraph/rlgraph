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

from yarl import backend
from yarl.utils import util
from yarl.components import Component
from yarl.spaces import ContainerSpace, Tuple

if backend == "tf":
    import tensorflow as tf


class Distribution(Component):
    """
    A distribution wrapper class that can incorporate a backend-specific distribution object that gets its parameters
    from an external source (e.g. a NN).

    API:
    ins:
        parameters (numeric): The parameters of the distribution (e.g. mean and variance for a Gaussian).
            The Space of parameters must have a batch-rank.
        Optional:
            max_likelihood (bool): Whether to sample or to get the max-likelihood value (deterministic) when
                using the "draw" out-Socket. This Socket is optional and can be switched on via the constructor parameter:
                "expose_draw"=True.
    outs:
        sample_stochastic (numeric): Returns a stochastic sample from the distribution.
        sample_deterministic (numeric): Returns the max-likelihood value (deterministic) from the distribution.
        entropy (float): The entropy value of the distribution.
        Optional:
            draw (numeric): Draws a sample from the distribution (if max_likelihood is True, this is will be
                a deterministic draw, otherwise a stochastic sample). This Socket is optional and can be switched on via
                the constructor parameter: "expose_draw"=True. By default, this Socket is not exposed.
    """
    def __init__(self, expose_draw=False, scope="distribution", **kwargs):
        """
        Args:
            expose_draw (bool): Whether this Component should expose an out-Socket named "draw"
                (needing also a 'max_likelihood' in-Socket). This additional out-Socket either returns a
                stochastic or a deterministic sample from the distribution, depending on the provided
                "max_likelihood" in-Socket bool value.
                Default: False.
        """
        super(Distribution, self).__init__(scope=scope, flatten_ops=kwargs.pop("flatten_ops", False), **kwargs)

        # Define a generic Distribution interface.
        self.define_inputs("parameters")
        self.define_outputs("sample_stochastic", "sample_deterministic", "entropy")
        # "distribution" will be an internal Socket used to connect the GraphFunctions with each other.
        self.add_graph_fn("parameters", "distribution", self._graph_fn_parameterize)
        self.add_graph_fn("distribution", "sample_stochastic", self._graph_fn_sample_stochastic)
        self.add_graph_fn("distribution", "sample_deterministic", self._graph_fn_sample_deterministic)
        self.add_graph_fn("distribution", "entropy", self._graph_fn_entropy)
        self.add_graph_fn(["distribution_a", "distribution_b"], "kl_divergence", self._graph_fn_kl_divergence)

        # If we need the flexible out-Socket "draw", add it here and connect it.
        if expose_draw is True:
            self.define_inputs("max_likelihood")
            self.define_outputs("draw")
            self.add_graph_fn(["distribution", "max_likelihood"], "draw", self._graph_fn_draw)

    def check_input_spaces(self, input_spaces):
        in_space = input_spaces["parameters"]
        # Must not be ContainerSpace (not supported yet for Distributions, doesn't seem to make sense).
        assert not isinstance(in_space, ContainerSpace), "ERROR: Cannot handle container input Spaces " \
                                                         "in distribution '{}' (atm; may soon do)!".format(self.name)

    def _graph_fn_parameterize(self, *parameters):
        """
        Parameterizes this distribution (normally from an NN-output vector). Returns
        the backend-distribution object (a DataOp).

        Args:
            *parameters (DataOp): The input(s) used to parameterize this distribution. This is normally a cleaned up
                single NN-output that (e.g.: the two values for mean and variance for a univariate Gaussian
                distribution).

        Returns:
            DataOp: The parameterized backend-specific distribution object.
        """
        raise NotImplementedError

    def _graph_fn_draw(self, distribution, max_likelihood):
        """
        Takes a sample from the (already parameterized) distribution. The parameterization also includes a possible
        batch size.

        Args:
            distribution (DataOp): The (already parameterized) backend-specific distribution DataOp to use for
                sampling. This is simply the output of `self._graph_fn_parameterize`.
            max_likelihood (bool): Whether to return the maximum-likelihood result, instead of a random sample.
                Can be used to pick deterministic actions from discrete ("greedy") or continuous (mean-value)
                distributions.

        Returns:
            DataOp: The taken sample(s).
        """
        if backend == "tf":
            return tf.cond(
                pred=max_likelihood,
                true_fn=lambda: self._graph_fn_sample_deterministic(distribution),
                false_fn=lambda: self._graph_fn_sample_stochastic(distribution)
            )

    def _graph_fn_sample_deterministic(self, distribution):
        """
        Returns the maximum-likelihood value for a given distribution.

        Args:
            distribution (DataOp): The (already parameterized) backend-specific distribution whose max-likelihood value
                to calculate. This is simply the output of `self._graph_fn_parameterize`.

        Returns:
            DataOp: The max-likelihood value.
        """
        raise NotImplementedError

    @staticmethod
    def _graph_fn_sample_stochastic(distribution):
        """
        Returns an actual sample for a given distribution.

        Args:
            distribution (DataOp): The (already parameterized) backend-specific distribution from which a sample
                should be drawn. This is simply the output of `self._graph_fn_parameterize`.

        Returns:
            DataOp: The drawn sample.
        """
        return distribution.sample()

    @staticmethod
    def _graph_fn_entropy(distribution):
        """
        Returns the DataOp holding the entropy value of the distribution.

        Args:
            distribution (DataOp): The (already parameterized) backend-specific distribution whose entropy to
                calculate. This is simply the output of `self._graph_fn_parameterize`.

        Returns:
            DataOp: The distribution's entropy.
        """
        return distribution.entropy()

    @staticmethod
    def _graph_fn_kl_divergence(distribution_a, distribution_b):
        """
        Kullback-Leibler divergence between two distribution objects.

        Args:
            distribution_a (tf.Distribution): A Distribution object.
            distribution_b (tf.Distribution): A distribution object.

        Returns:
            DataOp: (batch-wise) KL-divergence between the two distributions.
        """
        if backend == "tf":
            return tf.distributions.kl_divergence(
                distribution_a=distribution_a,
                distribution_b=distribution_b,
                allow_nan_stats=True,
                name=None
            )


class Bernoulli(Distribution):
    """
    A Bernoulli distribution object defined by a single value p, the probability for True (rather than False).
    """
    def __init__(self, scope="bernoulli", **kwargs):
        super(Bernoulli, self).__init__(scope=scope, **kwargs)

    def _graph_fn_parameterize(self, prob):
        """
        Args:
            prob (DataOp): The p value (probability that distribution returns True).
        """
        # TODO: Move this into some sort of NN-output-cleanup component. This is repetitive stuff.
        ## Clamp raw_input between 0 and 1 to make it interpretable as a probability.
        #p = tf.sigmoid(x=flat_input)
        ## Clamp to avoid 0.0 or 1.0 probabilities (adds numerical stability).
        #p = tf.clip_by_value(p, clip_value_min=SMALL_NUMBER, clip_value_max=(1.0 - SMALL_NUMBER))
        if backend == "tf":
            return tf.distributions.Bernoulli(probs=prob, dtype=util.dtype("bool"))

    def _graph_fn_sample_deterministic(self, distribution):
        return distribution.prob(True) >= 0.5


class Categorical(Distribution):
    """
    A categorical distribution object defined by a n values {p0, p1, ...} that add up to 1, the probabilities
    for picking one of the n categories.
    """
    def __init__(self, scope="categorical", **kwargs):
        super(Categorical, self).__init__(scope=scope, **kwargs)

    def _graph_fn_parameterize(self, probs):
        if backend == "tf":
            return tf.distributions.Categorical(probs=probs, dtype=util.dtype("int"))

    def _graph_fn_sample_deterministic(self, distribution):
        if backend == "tf":
            return tf.argmax(input=distribution.probs, axis=-1, output_type=util.dtype("int"))


class Normal(Distribution):
    """
    A categorical distribution object defined by a n values {p0, p1, ...} that add up to 1, the probabilities
    for picking one of the n categories.
    """
    def __init__(self, scope="normal", **kwargs):
        # Do not flatten incoming DataOps as we need more than one parameter in our parameterize graph_fn.
        super(Normal, self).__init__(scope=scope, **kwargs)

    def check_input_spaces(self, input_spaces):
        # Must be a Tuple of len 2 (loc and scale).
        in_space = input_spaces["parameters"]
        assert isinstance(in_space, Tuple) and len(in_space) == 2,\
            "ERROR: Normal Distribution ({}) needs an incoming Tuple with len=2!".format(self.name)

    def _graph_fn_parameterize(self, loc_and_scale):
        if backend == "tf":
            return tf.distributions.Normal(loc=loc_and_scale[0], scale=loc_and_scale[1])

    def _graph_fn_sample_deterministic(self, distribution):
        return distribution.mean()


class Beta(Distribution):
    """
    A Beta distribution is defined on the interval [0, 1] and parameterized by shape parameters
    alpha and beta (also called concentration parameters).
    """
    def __init__(self, scope="beta", **kwargs):
        # Do not flatten incoming DataOps as we need more than one parameter in our parameterize graph_fn.
        super(Beta, self).__init__(scope=scope, **kwargs)

    def check_input_spaces(self, input_spaces):
        # Must be a Tuple of len 2 (loc and scale).
        in_space = input_spaces["parameters"]
        assert isinstance(in_space, Tuple) and len(in_space) == 2,\
            "ERROR: Beta Distribution ({}) needs an incoming Tuple with len=2!".format(self.name)

    def _graph_fn_parameterize(self, concentration_parameters):
        if backend == "tf":
            return tf.distributions.Beta(
                concentration0=concentration_parameters[0],
                concentration1=concentration_parameters[1]
            )

    def _graph_fn_sample_deterministic(self, distribution):
        return distribution.mean()