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

from rlgraph import get_backend
from rlgraph.utils.decorators import rlgraph_api, graph_fn
from rlgraph.components import Component
from rlgraph.spaces import ContainerSpace

if get_backend() == "tf":
    import tensorflow as tf


class Distribution(Component):
    """
    A distribution wrapper class that can incorporate a backend-specific distribution object that gets its parameters
    from an external source (e.g. a NN).

    API:
        get_distribution(parameters): The backend-specific distribution object.
        sample_stochastic(parameters): Returns a stochastic sample from the distribution.
        sample_deterministic(parameters): Returns the max-likelihood value (deterministic) from the distribution.

        draw(parameters, max_likelihood): Draws a sample from the distribution (if `max_likelihood` is True,
            this is will be a deterministic draw, otherwise a stochastic sample).

        entropy(parameters): The entropy value of the distribution.
        log_prob(parameters): The log probabilities for given values.

        kl_divergence(parameters, other_parameters): The Kullback-Leibler Divergence between a Distribution and
            another one.
    """
    def __init__(self, scope="distribution", **kwargs):
        """
        Keyword Args:
            seed (Optional[int]): An optional random seed to use when sampling stochastically.
        """
        self.seed = kwargs.pop("seed", None)
        super(Distribution, self).__init__(scope=scope, **kwargs)

        # TEST: Try setting components that don't have variables to input_complete AND variable_complete right away.
        self.input_complete = True
        self.variable_complete = True
        # END: TEST

        # For define-by-run to avoid creating new objects when calling `get_distribution`.
        self.dist_object = None

    def check_input_spaces(self, input_spaces, action_space=None):
        ## The first arg of all API-methods is always the distribution parameters. Check them for ContainerSpaces.
        #for key in ["sample_stochastic", "sample_deterministic", "draw",
        #            "entropy", "log_prob", "kl_divergence"]:
        for key in ["distribution", "distribution_b"]:
            if key in input_spaces:
                parameter_space = input_spaces[key]
                # Must not be ContainerSpace (not supported yet for Distributions, doesn't seem to make sense).
                assert not isinstance(parameter_space, ContainerSpace),\
                    "ERROR: Cannot handle container parameter Spaces in distribution '{}' " \
                    "(atm; may soon do)!".format(self.name)

    # Now use that API-method to get the distribution object to implement all other API-methods.
    def sample_stochastic(self, parameters):
        distribution = self._graph_fn_get_distribution(parameters)
        return self._graph_fn_sample_stochastic(distribution)

    def sample_deterministic(self, parameters):
        distribution = self._graph_fn_get_distribution(parameters)
        return self._graph_fn_sample_deterministic(distribution)

    def draw(self, parameters, max_likelihood=True):
        distribution = self._graph_fn_get_distribution(parameters)
        return self._graph_fn_draw(distribution, max_likelihood)

    def entropy(self, parameters):
        distribution = self._graph_fn_get_distribution(parameters)
        return self._graph_fn_entropy(distribution)

    @rlgraph_api(must_be_complete=False)
    def log_prob(self_, parameters, values):
        distribution = self_._graph_fn_get_distribution(parameters)
        return self_._graph_fn_log_prob(distribution, values)

    @rlgraph_api(must_be_complete=False)
    def kl_divergence(self_, parameters, other_parameters):
        distribution = self_._graph_fn_get_distribution(parameters)
        other_distribution = self_._graph_fn_get_distribution(other_parameters)
        return self_._graph_fn_kl_divergence(distribution, other_distribution)

    @rlgraph_api
    def _graph_fn_get_distribution(self, *parameters):
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

    @graph_fn
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
        if get_backend() == "tf":
            return tf.cond(
                pred=max_likelihood,
                true_fn=lambda: self._graph_fn_sample_deterministic(distribution),
                false_fn=lambda: self._graph_fn_sample_stochastic(distribution)
            )
        elif get_backend() == "pytorch":
            if max_likelihood:
                return self._graph_fn_sample_deterministic(distribution)
            else:
                self._graph_fn_sample_stochastic(distribution)

    @graph_fn
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

    @graph_fn
    def _graph_fn_sample_stochastic(self, distribution):
        """
        Returns an actual sample for a given distribution.

        Args:
            distribution (DataOp): The (already parameterized) backend-specific distribution from which a sample
                should be drawn. This is simply the output of `self._graph_fn_parameterize`.

        Returns:
            DataOp: The drawn sample.
        """
        return distribution.sample(seed=self.seed)

    @graph_fn
    def _graph_fn_log_prob(self, distribution, values):
        """
        Probability density/mass function.

        Args:
            distribution (DataOp): The (already parameterized) backend-specific distribution for which the log
                probabilities should be calculated. This is simply the output of `self._graph_fn_parameterize`.
            values (SingleDataOp): Values for which to compute the log probabilities given `distribution`.

        Returns:
            DataOp: The log probability of the given values.
        """
        return distribution.log_prob(value=values)

    @graph_fn
    def _graph_fn_entropy(self, distribution):
        """
        Returns the DataOp holding the entropy value of the distribution.

        Args:
            distribution (DataOp): The (already parameterized) backend-specific distribution whose entropy to
                calculate. This is simply the output of `self._graph_fn_parameterize`.

        Returns:
            DataOp: The distribution's entropy.
        """
        return distribution.entropy()

    @graph_fn
    def _graph_fn_kl_divergence(self, distribution, distribution_b):
        """
        Kullback-Leibler divergence between two distribution objects.

        Args:
            distribution (tf.Distribution): The (already parameterized) backend-specific distribution 1.
            distribution_b (tf.Distribution): The other distribution object.

        Returns:
            DataOp: (batch-wise) KL-divergence between the two distributions.
        """
        if get_backend() == "tf":
            return tf.no_op()
            # TODO: never tested. tf throws error: NotImplementedError: No KL(distribution_a || distribution_b) registered for distribution_a type Bernoulli and distribution_b type ndarray
            #return tf.distributions.kl_divergence(
            #    distribution_a=distribution_a,
            #    distribution_b=distribution_b,
            #    allow_nan_stats=True,
            #    name=None
            #)
