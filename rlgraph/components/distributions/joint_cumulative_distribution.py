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

from __future__ import absolute_import, division, print_function

from rlgraph import get_backend
from rlgraph.components.distributions.distribution import Distribution
from rlgraph.components.layers.preprocessing.reshape import ReShape
from rlgraph.utils.decorators import rlgraph_api, graph_fn
from rlgraph.utils.ops import flatten_op, FlattenedDataOp

if get_backend() == "tf":
    import tensorflow as tf
elif get_backend() == "pytorch":
    import torch


class JointCumulativeDistribution(Distribution):
    """
    A joint cumulative distribution consisting of an arbitrarily nested container of n sub-distributions
    assumed to be all independent(!) of each other, such that:
    For e.g. n=2 and random variables X and Y: P(X and Y) = P(X)*P(Y) for all x and y.
    - Sampling returns a ContainerDataOp.
    - log_prob returns the sum of all single log prob terms (joint log prob).
    - entropy returns the sum of all single entropy terms (joint entropy).
    """
    def __init__(self, distribution_specs, scope="joint-cumulative-distribution", **kwargs):
        """
        Args:
            distribution_specs (dict): Dict with flat-keys containing the specifications of the single
                sub-distributions.
        """
        super(JointCumulativeDistribution, self).__init__(scope=scope, **kwargs)

        # Create the flattened sub-distributions and add them.
        self.flattened_sub_distributions = \
            {flat_key: Distribution.from_spec(spec, scope="sub-distribution-{}".format(i))
             for i, (flat_key, spec) in enumerate(distribution_specs.items())
            }
        self.flattener = ReShape(flatten=True)

        self.add_components(self.flattener, *list(self.flattened_sub_distributions.values()))

    @rlgraph_api
    def sample_deterministic(self, parameters):
        return self._graph_fn_sample_deterministic(parameters)

    @rlgraph_api
    def sample_stochastic(self, parameters):
        return self._graph_fn_sample_stochastic(parameters)

    @rlgraph_api
    def draw(self, parameters, deterministic=True):
        return self._graph_fn_draw(parameters, deterministic)

    @rlgraph_api
    def sample_and_log_prob(self, parameters, deterministic=True):
        #distribution = self.get_distribution(parameters)
        actions = self._graph_fn_draw(parameters, deterministic)
        log_probs = self._graph_fn_log_prob(parameters, actions)
        return actions, log_probs

    #@rlgraph_api
    #def entropy(self, parameters):
    #    return self._graph_fn_entropy(parameters)

    @rlgraph_api
    def log_prob(self, parameters, values):
        """
        Override log_prob API as we have to add all the resulting log-probs together
        (joint log-prob of individual ones).
        """
        #distributions = self.get_distribution(parameters)
        all_log_probs = self._graph_fn_log_prob(parameters, values)
        return self._graph_fn_reduce_over_sub_distributions(all_log_probs)

    #@rlgraph_api(must_be_complete=False)
    #def kl_divergence(self, parameters, other_parameters):
    #    distribution = self.get_distribution(parameters)
    #    other_distribution = self.get_distribution(other_parameters)
    #    return self._graph_fn_kl_divergence(distribution, other_distribution)

    # Flatten only alongside `self.flattened_sub_distributions`, not any further.
    @rlgraph_api(flatten_ops="flattened_sub_distributions", split_ops=True, add_auto_key_as_first_param=True, ok_to_overwrite=True)
    def _graph_fn_get_distribution(self, key, parameters):
        return self.flattened_sub_distributions[key].get_distribution(parameters)

    @graph_fn(flatten_ops="flattened_sub_distributions")
    def _graph_fn_sample_deterministic(self, parameters):
        ret = {}
        for key in parameters:
            ret[key] = self.flattened_sub_distributions[key].sample_deterministic(parameters[key])
        return FlattenedDataOp(ret)

    @graph_fn(flatten_ops="flattened_sub_distributions")
    def _graph_fn_sample_stochastic(self, parameters):
        ret = {}
        for key in parameters:
            ret[key] = self.flattened_sub_distributions[key].sample_stochastic(parameters[key])
        return FlattenedDataOp(ret)

    @graph_fn(flatten_ops="flattened_sub_distributions")
    def _graph_fn_draw(self, parameters, deterministic):
        ret = {}
        for key in parameters:
            ret[key] = self.flattened_sub_distributions[key].draw(parameters[key], deterministic)
        return FlattenedDataOp(ret)

    # Flatten only alongside `self.flattened_sub_distributions`, not any further.
    @graph_fn(flatten_ops="flattened_sub_distributions")
    def _graph_fn_log_prob(self, parameters, values):
        ret = {}
        for key in parameters:
            #d = self.flattened_sub_distributions[key].get_distribution(parameters[key])
            #return self.flattened_sub_distributions[key]._graph_fn_log_prob(distribution, values)
            ret[key] = self.flattened_sub_distributions[key].log_prob(parameters[key], values[key])
        return FlattenedDataOp(ret)

    @graph_fn(flatten_ops=True)
    def _graph_fn_reduce_over_sub_distributions(self, log_probs):
        params_space = next(iter(flatten_op(self.api_method_inputs["parameters"]).values()))
        num_ranks_to_keep = (1 if params_space.has_batch_rank else 0) + (1 if params_space.has_time_rank else 0)
        log_probs_list = []
        if get_backend() == "tf":
            for log_prob in log_probs.values():
                # Reduce sum over all ranks to get the joint log llh.
                log_prob = tf.reduce_sum(log_prob, axis=list(range(len(log_prob.shape) - 1, num_ranks_to_keep - 1, -1)))
                log_probs_list.append(log_prob)
            return tf.reduce_sum(tf.stack(log_probs_list, axis=0), axis=0)

        elif get_backend() == "pytorch":
            for log_prob in log_probs.values():
                # Reduce sum over all ranks to get the joint log llh.
                log_prob = torch.sum(log_prob, dim=list(range(len(log_prob.shape) - 1, num_ranks_to_keep - 1, -1)))
                log_probs_list.append(log_prob)

            return torch.sum(torch.stack(log_probs_list, dim=0), dim=0)

    # Flatten only alongside `self.flattened_sub_distributions`, not any further.
    @graph_fn(flatten_ops="flattened_sub_distributions")
    def _graph_fn_entropy(self, distribution):
        params_space = next(iter(flatten_op(self.api_method_inputs["parameters"]).values()))
        num_ranks_to_keep = (1 if params_space.has_batch_rank else 0) + (1 if params_space.has_time_rank else 0)
        all_entropies = []
        if get_backend() == "tf":
            for key, distr in distribution.items():
                entropy = distr.entropy()
                # Reduce sum over all ranks to get the joint entropy.
                entropy = tf.reduce_sum(entropy, axis=list(range(len(entropy.shape) - 1, num_ranks_to_keep - 1, -1)))
                all_entropies.append(entropy)
            return tf.reduce_sum(tf.stack(all_entropies, axis=0), axis=0)

        elif get_backend() == "pytorch":
            for key, distr in distribution.items():
                entropy = distr.entropy()
                # Reduce sum over all ranks to get the joint log llh.
                entropy = torch.sum(entropy, dim=list(range(len(entropy.shape) - 1, num_ranks_to_keep - 1, -1)))
                all_entropies.append(entropy)

            # TODO: flatten all all_log_probs (or expand in last dim) so we can concat, then reduce_sum to get the joint probs.
            return torch.sum(torch.stack(all_entropies, dim=0), dim=0)
