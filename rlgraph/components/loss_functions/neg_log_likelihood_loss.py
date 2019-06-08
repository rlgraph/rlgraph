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
from rlgraph.components.distributions.distribution import Distribution
from rlgraph.components.loss_functions.supervised_loss_function import SupervisedLossFunction
from rlgraph.utils.decorators import rlgraph_api

if get_backend() == "tf":
    import tensorflow as tf


class NegativeLogLikelihoodLoss(SupervisedLossFunction):
    """
    Calculates the negative log-likelihood loss by passing the labels through a given distribution
    (parameterized by `predictions`) and inverting the sign.

    L(params,labels) = -log(Dparams.pdf(labels))
    Where:
        Dparams: Parameterized distribution object.
        pdf: Prob. density function of the distribution.
    """
    def __init__(self, distribution_spec, average_time_steps=False, scope="negative-log-likelihood-loss", **kwargs):
        """
        Args:
            average_time_steps (bool): Whether, if a time rank is given, to divide by th esequence lengths to get
                the mean or not (leave as sum).
        """
        super(NegativeLogLikelihoodLoss, self).__init__(scope=scope, **kwargs)

        self.distribution = Distribution.from_spec(distribution_spec)
        self.average_time_steps = average_time_steps

        self.add_components(self.distribution)

        #self.reduce_ranks = None

        self.time_rank = None
        self.time_major = None

    def check_input_spaces(self, input_spaces, action_space=None):
        in_space = input_spaces["labels"]

        #self.reduce_ranks = np.array(list(range(in_space.rank)))
        #if in_space.has_batch_rank:
        #    self.reduce_ranks += 1
        #if in_space.has_time_rank:
        #    self.reduce_ranks += 1

        self.time_rank = in_space.has_time_rank
        self.time_major = in_space.time_major

    @rlgraph_api
    def _graph_fn_loss_per_item(self, parameters, labels, sequence_length=None, time_percentage=None):
        """
        Args:
            parameters (SingleDataOp): Output parameters for a distribution.
            labels (SingleDataOp): Labels that will be passed through the pdf function of the distribution.
            sequence_length (SingleDataOp): The lengths of each sequence (if applicable) in the given batch.
            time_percentage (Optional[SingleDataOp]): The time_percentage of the update. May be used e.g. for decaying
                some weight parameter.

        Returns:
            SingleDataOp: The loss values vector (one single value for each batch item).
        """
        batch_rank = 0 if self.time_major is False else 1
        #time_rank = 0 if batch_rank == 1 else 1

        params_space = next(iter(self.api_method_inputs["parameters"].flatten().values()))
        num_ranks_to_keep = 2 if params_space.has_time_rank else 1

        if get_backend() == "tf":
            # Get the distribution's log-likelihood for the labels, given the parameterized distribution.
            neg_log_likelihood = -self.distribution.log_prob(parameters, labels)

            # If necessary, reduce over all non-batch/non-time ranks.
            neg_log_likelihood = tf.reduce_sum(
                neg_log_likelihood,
                axis=list(range(len(neg_log_likelihood.shape) - 1, num_ranks_to_keep - 1, -1))
            )

            # TODO: Here, we use no time-decay and just sum up the valid time-steps.
            #if sequence_length is not None:
            #    max_time_steps = tf.cast(tf.shape(labels)[time_rank], dtype=tf.float32)
            #    sequence_mask = tf.sequence_mask(sequence_length, max_time_steps, dtype=tf.float32)
            #    neg_log_likelihoods = tf.multiply(neg_log_likelihoods, sequence_mask)
            #    # Reduce away the time-rank.
            #    neg_log_likelihoods = tf.reduce_sum(neg_log_likelihoods, axis=time_rank)
            #    # Average?
            #    if self.average_time_steps is True:
            #        neg_log_likelihoods = tf.divide(neg_log_likelihoods, tf.cast(sequence_length, dtype=tf.float32))
            ## Reduce away the time-rank.
            #elif self.time_rank:
            #    neg_log_likelihoods = tf.reduce_mean(neg_log_likelihoods, axis=time_rank)

            neg_log_likelihood._batch_rank = batch_rank

            return neg_log_likelihood
