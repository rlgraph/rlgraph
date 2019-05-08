# Copyright 2018/2019 ducandu GmbH, All Rights Reserved.
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

from rlgraph import get_backend
from rlgraph.components.action_adapters.normal_distribution_adapter import NormalDistributionAdapter
from rlgraph.spaces import FloatBox
from rlgraph.utils.decorators import graph_fn
from rlgraph.utils.ops import DataOpDict, DataOpTuple
from rlgraph.utils.util import SMALL_NUMBER

if get_backend() == "tf":
    import tensorflow as tf


class NormalMixtureDistributionAdapter(NormalDistributionAdapter):
    def __init__(self, action_space, num_mixtures=1, scope="normal-mixture-adapter", **kwargs):
        """
        Args:
            num_mixtures (int): The mixture's size (number of sub-distributions to categorically sample from).
                Default: 1 (no mixture).
        """
        self.num_mixtures = num_mixtures
        super(NormalMixtureDistributionAdapter, self).__init__(action_space, scope=scope, **kwargs)

    def get_units_and_shape(self):
        if self.num_mixtures == 1:
            return super(NormalMixtureDistributionAdapter, self).get_units_and_shape()

        new_shape = list(self.action_space.get_shape(with_category_rank=True))
        new_shape = tuple(new_shape[:-1] + [self.num_mixtures + self.num_mixtures * 2 * new_shape[-1]])
        last_dim = self.action_space.get_shape()[-1]
        units = self.num_mixtures + self.num_mixtures * last_dim * 2
        return units, new_shape

    @graph_fn
    def _graph_fn_get_parameters_from_adapter_outputs(self, adapter_outputs):
        """
        Creates properties/parameters and log-probs from some reshaped output.

        Args:
            logits (SingleDataOp): The output of some layer that is already reshaped
                according to our action Space.

        Returns:
            tuple (2x SingleDataOp):
                parameters (DataOp): The parameters, ready to be passed to a Distribution object's
                    get_distribution API-method (usually some probabilities or loc/scale pairs).
                log_probs (DataOp): Simply the log(parameters).
        """
        # Shortcut: If no mixture distribution, let ActionAdapter parent deal with everything.
        if self.num_mixtures == 1:
            return super(NormalMixtureDistributionAdapter, self)._graph_fn_get_parameters_from_adapter_outputs(
                adapter_outputs
            )

        parameters = None
        log_probs = None

        if get_backend() == "tf":
            # Continuous actions.
            # For now, assume unbounded outputs.
            assert isinstance(self.action_space, FloatBox) and self.action_space.unbounded

            parameters = DataOpDict()
            log_probs = DataOpDict()

            # Nodes encode the following:
            # - [num_mixtures] (for categorical)
            # - []

            # Unbounded -> Mixture Multivariate Normal distribution.
            last_dim = self.action_space.get_shape()[-1]
            categorical, means, log_sds = tf.split(adapter_outputs, num_or_size_splits=[
                self.num_mixtures, self.num_mixtures * last_dim, self.num_mixtures * last_dim
            ], axis=-1)

            # Parameterize the categorical distribution, which will pick one of the mixture ones.
            parameters["categorical"] = tf.maximum(x=tf.nn.softmax(logits=categorical, axis=-1), y=SMALL_NUMBER)
            parameters["categorical"]._batch_rank = 0
            # Log probs.
            log_probs["categorical"] = tf.log(x=parameters["categorical"])
            log_probs["categorical"]._batch_rank = 0

            # Turn log sd into sd to ascertain always positive stddev values.
            sds = tf.exp(log_sds)
            log_means = tf.log(means)

            # Split into one for each item in the Mixture.
            means = tf.split(means, num_or_size_splits=self.num_mixtures, axis=-1)
            log_means = tf.split(log_means, num_or_size_splits=self.num_mixtures, axis=-1)
            sds = tf.split(sds, num_or_size_splits=self.num_mixtures, axis=-1)
            log_sds = tf.split(log_sds, num_or_size_splits=self.num_mixtures, axis=-1)

            # Store each mixture item's parameters in DataOpDict.
            for i in range(self.num_mixtures):
                mean = means[i]
                mean._batch_rank = 0
                sd = sds[i]
                sd._batch_rank = 0

                log_mean = log_means[i]
                log_mean._batch_rank = 0
                log_sd = log_sds[i]
                log_sd._batch_rank = 0

                parameters["parameters{}".format(i)] = DataOpTuple([mean, sd])
                log_probs["parameters{}".format(i)] = DataOpTuple([log_mean, log_sd])

        return parameters, log_probs
