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

from math import log

from rlgraph import get_backend
from rlgraph.components.action_adapters import ActionAdapter
from rlgraph.utils.decorators import graph_fn
from rlgraph.utils.util import SMALL_NUMBER


if get_backend() == "tf":
    import tensorflow as tf
elif get_backend() == "pytorch":
    import torch


class BetaDistributionAdapter(ActionAdapter):
    """Action adapter for the Beta distribution"""

    def get_units_and_shape(self):
        units = 2 * self.action_space.flat_dim  # Those two dimensions are the mean and log sd
        # Add moments (2x for each action item).
        if self.action_space.shape == ():
            new_shape = (2,)
        else:
            new_shape = tuple(list(self.action_space.shape[:-1]) + [self.action_space.shape[-1] * 2])
        return units, new_shape

    @graph_fn
    def _graph_fn_get_parameters_log_probs(self, logits):
        parameters = None
        log_probs = None

        if get_backend() == "tf":
            # Stabilize both alpha and beta (currently together in parameters).
            parameters = tf.clip_by_value(
                logits, clip_value_min=log(SMALL_NUMBER), clip_value_max=-log(SMALL_NUMBER)
            )
            parameters = tf.log((tf.exp(parameters) + 1.0)) + 1.0
            parameters._batch_rank = 0
            log_probs = tf.log(parameters)
            log_probs._batch_rank = 0

        elif get_backend() == "pytorch":
            parameters = torch.clamp(
                logits, min=log(SMALL_NUMBER), max=-log(SMALL_NUMBER)
            )
            parameters = torch.log((torch.exp(parameters) + 1.0)) + 1.0
            log_probs = torch.log(parameters)

        return parameters, log_probs
