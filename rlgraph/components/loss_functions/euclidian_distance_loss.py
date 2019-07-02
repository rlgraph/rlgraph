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

import numpy as np

from rlgraph import get_backend
from rlgraph.components.loss_functions.supervised_loss_function import SupervisedLossFunction
from rlgraph.spaces.bool_box import BoolBox
from rlgraph.utils.decorators import rlgraph_api

if get_backend() == "tf":
    import tensorflow as tf


class EuclidianDistanceLoss(SupervisedLossFunction):
    """
    Calculates the loss between two vectors (prediction and label) via their Euclidian distance:
    d(v,w) = SQRT(SUMi( (vi - wi)² ))
    """
    def __init__(self, time_steps=None, scope="euclidian-distance", **kwargs):
        """
        Args:
            time_steps (Optional[int]): If given, reduce-sum linearly over this many timesteps with weights going
                from 0.0 (first time-step) to 1.0 (last-timestep).
        """
        super(EuclidianDistanceLoss, self).__init__(scope=scope, **kwargs)

        self.time_steps = time_steps
        self.reduce_ranks = None

        self.time_rank = None
        self.time_major = None

        self.is_bool = None

    def check_input_spaces(self, input_spaces, action_space=None):
        in_space = input_spaces["labels"]
        self.is_bool = isinstance(in_space, BoolBox)  # Need to cast (to 0.0 and 1.0) in graph_fn?
        self.reduce_ranks = np.array(list(range(in_space.rank)))
        if in_space.has_batch_rank:
            self.reduce_ranks += 1
        if in_space.has_time_rank:
            self.reduce_ranks += 1

        self.time_rank = in_space.has_time_rank
        self.time_major = in_space.time_major

    @rlgraph_api
    def _graph_fn_loss_per_item(self, parameters, labels, sequence_length=None, time_percentage=None):
        """
        Euclidian distance loss.

        Args:
            parameters (SingleDataOp): Output predictions.
            labels (SingleDataOp): Labels.
            sequence_length (SingleDataOp): The lengths of each sequence (if applicable) in the given batch.

        Returns:
            SingleDataOp: The loss values vector (one single value for each batch item).
        """
        batch_rank = 0 if self.time_major is False else 1
        time_rank = 0 if batch_rank == 1 else 1

        if get_backend() == "tf":
            # Reduce over last rank (vector axis) and take the square root.
            if self.is_bool:
                labels = tf.cast(labels, tf.float32)
                parameters = tf.cast(parameters, tf.float32)
            euclidian_distance = tf.square(tf.subtract(parameters, labels))
            euclidian_distance = tf.reduce_sum(euclidian_distance, axis=self.reduce_ranks)
            euclidian_distance = tf.sqrt(euclidian_distance)

            # TODO: Make it possible to customize the time-step decay (or increase?) behavior.
            # Weight over time-steps (linearly decay weighting over time rank, cutting out entirely values past the
            # sequence length).
            if sequence_length is not None:
                max_time_steps = tf.cast(tf.shape(labels)[time_rank], dtype=tf.float32)
                sequence_mask = tf.sequence_mask(sequence_length, max_time_steps, dtype=tf.float32)
                sequence_decay = tf.expand_dims(
                    tf.range(start=1.0, limit=0.0, delta=-1.0 / max_time_steps, dtype=tf.float32), axis=batch_rank
                )
                weighting = sequence_mask * sequence_decay
                euclidian_distance = tf.multiply(euclidian_distance, weighting)
                # Reduce away the time-rank.
                euclidian_distance = tf.reduce_sum(euclidian_distance, axis=time_rank)
                euclidian_distance = tf.divide(euclidian_distance, tf.cast(sequence_length, dtype=tf.float32))
            else:
                # Reduce away the time-rank.
                if hasattr(parameters, "_time_rank"):
                    euclidian_distance = tf.reduce_mean(euclidian_distance, axis=time_rank)

            return euclidian_distance
