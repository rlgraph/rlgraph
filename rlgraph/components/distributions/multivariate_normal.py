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

from rlgraph import get_backend
from rlgraph.utils.decorators import rlgraph_api, graph_fn
from rlgraph.components.distributions.distribution import Distribution

if get_backend() == "tf":
    import tensorflow as tf
    #import tensorflow_probability as tfp
elif get_backend() == "pytorch":
    import torch


class MultivariateNormal(Distribution):
    """
    A multivariate Gaussian distribution.
    """
    def __init__(self, num_events=2, parameterize_via_diagonal=True,
                 parameterize_via_covariance=False, scope="multivariate-normal", **kwargs):
        """
        Args:
            num_events (int): The number of events in the multivariate Gaussian. Default: Bivariate (2).
            parameterize_via_diagonal (bool): Whether we are parameterizing via the diagonal stddev values.
                Note that
            parameterize_via_covariance (bool): Whether we are parameterizing via the full covariance values.
        """
        super(MultivariateNormal, self).__init__(scope=scope, **kwargs)
        self.num_events = num_events
        self.parameterize_via_diagonal = parameterize_via_diagonal
        self.parameterize_via_covariance = parameterize_via_covariance
        assert self.parameterize_via_diagonal != self.parameterize_via_covariance, \
            "ERROR: Exactly one of `parameterize_via_diagonal` and `parameterize_via_covariance` must be True!"

    def check_input_spaces(self, input_spaces, action_space=None):
        # Must be a Tuple of len 2 (loc and scale).
        in_space = input_spaces["parameters"]
        if self.parameterize_via_diagonal:
            # Make sure input parameters has an even last rank for splitting into mean/stddev parameter values.
            assert in_space.shape[-1] % 2 == 0,\
                "ERROR: `parameters` in_space must have an even numbered last rank (mean/stddev split)!"
        # TODO: support parameterization through full covariance matrix.
        #else:
        #    # Make sure input parameters has an even last rank for splitting into mean/stddev parameter values.
        #    assert in_space.shape[-2:-1] == (self.num_events, 1 + self.num_events),\
        #        "ERROR: `parameters` in_space must have a second last rank of {} and a last rank of " \
        #        "{}!".format(self.num_events, self.num_events + 1)

        # Optional: must_be_complete=False.
        #if "nn_output" in input_spaces:
        #    nn_output_space = input_spaces["nn_output"]
        #    # Make sure nn_output has the correct length.
        #    assert nn_output_space[-1].shape == self.num_events * (self.num_events + 1),\
        #        "ERROR: `nn_output` in_space must be of length {}!".format(self.num_events * (self.num_events + 1))

    @rlgraph_api
    def _graph_fn_get_distribution(self, parameters):
        if get_backend() == "tf":
            if self.parameterize_via_diagonal:
                mean, scale_diag = tf.split(parameters, num_or_size_splits=2, axis=-1)
                return tfp.distributions.MultivariateNormalDiag(
                    loc=mean, scale_diag=scale_diag
                )
            # TODO: support parameterization through full covariance matrix.
            #else:
            #    mean, covariance_matrix = tf.split(parameters, num_or_size_splits=[1, self.num_events], axis=-1)
            #    mean = tf.squeeze(mean, axis=-1)
            #    return tfp.distributions.MultivariateNormalFullCovariance(
            #        loc=mean, covariance_matrix=covariance_matrix
            #    )

    @graph_fn
    def _graph_fn_sample_deterministic(self, distribution):
            return distribution.mean()
