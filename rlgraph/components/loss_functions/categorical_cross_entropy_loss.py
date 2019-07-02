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
from rlgraph.components.loss_functions.supervised_loss_function import SupervisedLossFunction
from rlgraph.spaces import IntBox, FloatBox
from rlgraph.spaces.space_utils import sanity_check_space
from rlgraph.utils.decorators import rlgraph_api
from rlgraph.utils.util import SMALL_NUMBER

if get_backend() == "tf":
    import tensorflow as tf


class CategoricalCrossEntropyLoss(SupervisedLossFunction):

    def __init__(self, sparse=True, with_kl_regularizer=True, average_time_steps=False, scope="cross-entropy-loss",
                 **kwargs):
        """
        Args:
            sparse (bool): Whether we have sparse labels. Sparse labels can only assign one category to each
                sample, so labels are ints. If False, labels are already softmaxed categorical distribution probs
                OR simple logits.

            average_time_steps (bool): Whether, if a time rank is given, to divide by th esequence lengths to get
                the mean or not (leave as sum).
        """
        super(CategoricalCrossEntropyLoss, self).__init__(scope=scope, **kwargs)

        self.sparse = sparse
        self.with_kl_regularizer = with_kl_regularizer
        self.average_time_steps = average_time_steps
        #self.reduce_ranks = None

        #self.time_rank = None
        #self.time_major = None

        #self.is_bool = None

    def check_input_spaces(self, input_spaces, action_space=None):
        labels_space = input_spaces["labels"]
        if self.sparse is True:
            sanity_check_space(labels_space, allowed_types=IntBox, must_have_batch_rank=True)
        else:
            sanity_check_space(labels_space, allowed_types=FloatBox, must_have_batch_rank=True)

    @rlgraph_api
    def _graph_fn_loss_per_item(self, parameters, labels, sequence_length=None, time_percentage=None):
        """
        Supervised cross entropy classification loss.

        Args:
            parameters (SingleDataOp): The parameters output by a DistributionAdapter (before sampling from a
                possible distribution).

            labels (SingleDataOp): The corresponding labels (ideal probabilities) or int categorical labels.
            sequence_length (SingleDataOp[int]): The lengths of each sequence (if applicable) in the given batch.

            time_percentage (SingleDataOp[bool]): The time-percentage (0.0 to 1.0) with respect to the max number of
                timesteps.

        Returns:
            SingleDataOp: The loss values vector (one single value for each batch item).
        """
        if get_backend() == "tf":
            batch_rank = parameters._batch_rank
            time_rank = 0 if batch_rank == 1 else 1

            # TODO: This softmaxing is duplicate computation (waste) as `parameters` are already softmaxed.
            if self.sparse is True:
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=parameters)
            else:
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=parameters)

            # TODO: Make it possible to customize the time-step decay (or increase?) behavior.
            # Weight over time-steps (linearly decay weighting over time rank, cutting out entirely values past the
            # sequence length).
            if sequence_length is not None:
                # Add KL Divergence between given distribution and uniform.
                if self.with_kl_regularizer is True:
                    uniform_probs = tf.fill(tf.shape(parameters), 1.0 / float(parameters.shape.as_list()[-1]))
                    # Subtract KL-divergence from loss term such that
                    kl = - tf.reduce_sum(uniform_probs * tf.log((tf.maximum(parameters, SMALL_NUMBER)) / uniform_probs), axis=-1)
                    cross_entropy += kl

                max_time_steps = tf.cast(tf.shape(labels)[time_rank], dtype=tf.float32)
                sequence_mask = tf.sequence_mask(sequence_length, max_time_steps, dtype=tf.float32)
                # no sequence decay anymore (no one does this):
                # sequence_decay = tf.range(start=1.0, limit=0.0, delta=-1.0 / max_time_steps, dtype=tf.float32)
                # sequence_decay = tf.range(start=0.5, limit=1.0, delta=0.5 / max_time_steps, dtype=tf.float32)
                weighting = sequence_mask  # * sequence_decay
                cross_entropy = tf.multiply(cross_entropy, weighting)

                # Reduce away the time-rank.
                cross_entropy = tf.reduce_sum(cross_entropy, axis=time_rank)
                # Average?
                if self.average_time_steps is True:
                    cross_entropy = tf.divide(cross_entropy, tf.cast(sequence_length, dtype=tf.float32))
            else:
                # Reduce away the time-rank.
                if hasattr(parameters, "_time_rank"):
                    cross_entropy = tf.reduce_sum(cross_entropy, axis=time_rank)

            return cross_entropy
