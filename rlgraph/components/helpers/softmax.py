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

from math import log

from rlgraph import get_backend
from rlgraph.utils.util import SMALL_NUMBER
from rlgraph.components.component import Component

if get_backend() == "tf":
    import tensorflow as tf


class SoftMax(Component):
    """
    A simple softmax component that translates logits into probabilities (and log-probabilities).

    API:
        apply(logits) -> returns probabilities (softmaxed) and log-probabilities.
    """
    def __init__(self, scope="softmax", **kwargs):
        super(SoftMax, self).__init__(scope=scope, **kwargs)

        self.define_api_method("get_probabilities_and_log_probs", self._graph_fn_get_probabilities_and_log_probs)

    def _graph_fn_get_probabilities_and_log_probs(self, logits):
        """
        Creates properties/parameters and log-probs from some reshaped output.

        Args:
            logits (SingleDataOp): The (already reshaped) logits.

        Returns:
            tuple (2x SingleDataOp):
                probabilities (DataOp): The probabilities after softmaxing the logits.
                log_probs (DataOp): Simply the log(probabilities).
        """
        if get_backend() == "tf":
            #if isinstance(self.action_space, IntBox):

            # Translate logits into probabilities in a save way (SMALL_NUMBER trick).
            probabilities = tf.maximum(x=tf.nn.softmax(logits=logits, axis=-1), y=SMALL_NUMBER)
            # Log probs.
            log_probs = tf.log(x=probabilities)

            #elif isinstance(self.action_space, FloatBox):
            #    # Continuous actions.
            #    mean, log_sd = tf.split(value=logits, num_or_size_splits=2, axis=1)
            #    # Remove moments rank.
            #    mean = tf.squeeze(input=mean, axis=1)
            #    log_sd = tf.squeeze(input=log_sd, axis=1)

            #    # Clip log_sd. log(SMALL_NUMBER) is negative.
            #    log_sd = tf.clip_by_value(t=log_sd, clip_value_min=log(SMALL_NUMBER), clip_value_max=-log(SMALL_NUMBER))

            #    # Turn log sd into sd.
            #    sd = tf.exp(x=log_sd)

            #    parameters = DataOpTuple(mean, sd)
            #    log_probs = DataOpTuple(tf.log(x=mean), log_sd)
            #else:
            #    raise NotImplementedError

            return probabilities, log_probs
