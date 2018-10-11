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
from rlgraph.utils.decorators import rlgraph_api

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

    @rlgraph_api(must_be_complete=False)
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
            # Translate logits into probabilities in a save way (SMALL_NUMBER trick).
            probabilities = tf.maximum(x=tf.nn.softmax(logits=logits, axis=-1), y=SMALL_NUMBER)
            # Log probs.
            log_probs = tf.log(x=probabilities)

            return probabilities, log_probs
