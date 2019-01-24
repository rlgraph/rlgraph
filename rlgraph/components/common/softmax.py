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
from rlgraph.utils.decorators import rlgraph_api
from rlgraph.utils.util import SMALL_NUMBER

if get_backend() == "tf":
    import tensorflow as tf
elif get_backend() == "pytorch":
    import torch


class Softmax(Component):
    """
    A simple softmax component that produces probabilities from logits.
    """
    def __init__(self, scope="softmax", **kwargs):
        super(Softmax, self).__init__(scope=scope, **kwargs)

    @rlgraph_api(flatten_ops=True, split_ops=True)
    def _graph_fn_softmax(self, logits):
        """
        Creates properties/parameters and log-probs from some reshaped logits output.

        Args:
            logits (SingleDataOp): The output of some layer that is already reshaped.

        Returns:
            tuple:
                probabilities (DataOp): The probabilities, ready to be passed to a Distribution object's
                    get_distribution API-method (usually some probabilities or loc/scale pairs).

                log_probs (DataOp): Simply the log(probabilities).
        """
        if get_backend() == "tf":
            # Discrete actions.
            probabilities = tf.maximum(x=tf.nn.softmax(logits=logits, axis=-1), y=SMALL_NUMBER)
            # Log probs.
            log_probs = tf.log(x=probabilities)
            return probabilities, log_probs

        elif get_backend() == "pytorch":
            # Discrete actions.
            probabilities = torch.max(torch.softmax(logits, dim=-1), torch.tensor(SMALL_NUMBER))
            # Log probs.
            log_probs = torch.log(probabilities)

            return probabilities, log_probs
