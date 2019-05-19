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
from rlgraph import get_backend
from rlgraph.components.component import Component
from rlgraph.utils.decorators import rlgraph_api, graph_fn

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

        draw(parameters, deterministic): Draws a sample from the distribution (if `deterministic` is True,
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

        # For define-by-run to avoid creating new objects when calling `get_distribution`.
        self.dist_object = None

    def get_action_adapter_type(self):
        """Returns the type of the action adapter to be used for this distribution.

        Returns:
            Union[str, class]: The type of the action adapter
        """
        raise NotImplementedError

    # Now use that API-method to get the distribution object to implement all other API-methods.
    @rlgraph_api
    def sample_stochastic(self, parameters):
        distribution = self.get_distribution(parameters)
        return self._graph_fn_sample_stochastic(distribution)

    @rlgraph_api
    def sample_deterministic(self, parameters):
        distribution = self.get_distribution(parameters)
        return self._graph_fn_sample_deterministic(distribution)

    @rlgraph_api
    def draw(self, parameters, deterministic=True):
        distribution = self.get_distribution(parameters)
        return self._graph_fn_draw(distribution, deterministic)

    @rlgraph_api
    def sample_and_log_prob(self, parameters, deterministic=True):
        distribution = self.get_distribution(parameters)
        actions = self._graph_fn_draw(distribution, deterministic)
        log_probs = self._graph_fn_log_prob(distribution, actions)
        return actions, log_probs

    @rlgraph_api
    def entropy(self, parameters):
        distribution = self.get_distribution(parameters)
        return self._graph_fn_entropy(distribution)

    @rlgraph_api(must_be_complete=False)
    def log_prob(self, parameters, values):
        distribution = self.get_distribution(parameters)
        return self._graph_fn_log_prob(distribution, values)

    @rlgraph_api(must_be_complete=False)
    def kl_divergence(self, parameters, other_parameters):
        distribution = self.get_distribution(parameters)
        other_distribution = self._graph_fn_get_distribution(other_parameters)
        return self._graph_fn_kl_divergence(distribution, other_distribution)

    @rlgraph_api
    def _graph_fn_get_distribution(self, parameters):
        """
        Parameterizes this distribution (normally from an NN-output vector). Returns
        the backend-distribution object (a DataOp).

        Args:
            parameters (DataOp): The input(s) used to parameterize this distribution. This is normally a cleaned up
                single NN-output (e.g.: the two values for mean and variance for a univariate Gaussian
                distribution).

        Returns:
            DataOp: The parameterized backend-specific distribution object.
        """
        raise NotImplementedError

    @graph_fn
    def _graph_fn_draw(self, distribution, deterministic):
        """
        Takes a sample from the (already parameterized) distribution. The parameterization also includes a possible
        batch size.

        Args:
            distribution (DataOp): The (already parameterized) backend-specific distribution DataOp to use for
                sampling. This is simply the output of `self._graph_fn_parameterize`.

            deterministic (Union[bool,DataOp]): Whether to return the maximum-likelihood result, instead of a random
                sample. Can be used to pick deterministic actions from discrete ("greedy") or continuous (mean-value)
                distributions.

        Returns:
            DataOp: The taken sample(s).
        """
        # Fixed boolean input (not a DataOp/Tensor).
        if get_backend() == "pytorch" or isinstance(deterministic, (bool, np.ndarray)):
            if deterministic is True:
                return self._graph_fn_sample_deterministic(distribution)
            else:
                return self._graph_fn_sample_stochastic(distribution)

        # For static graphs, `deterministic` could be a tensor.
        if get_backend() == "tf":
            return tf.cond(
                pred=deterministic,
                true_fn=lambda: self._graph_fn_sample_deterministic(distribution),
                false_fn=lambda: self._graph_fn_sample_stochastic(distribution)
            )

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
        if get_backend() == "tf":
            return distribution.sample(seed=self.seed)
        elif get_backend() == "pytorch":
            return distribution.sample()

    @graph_fn
    def _graph_fn_log_prob(self, distribution, values):
        """
        Probability density/mass function.

        Args:
            distribution (DataOp): The (already parameterized) backend-specific distribution for which the log
                probabilities should be calculated. This is simply the output of `self._graph_fn_get_distribution`.

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
