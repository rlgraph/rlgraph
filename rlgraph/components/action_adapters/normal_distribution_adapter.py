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
from rlgraph.utils.ops import DataOpTuple
from rlgraph.utils.util import SMALL_NUMBER

if get_backend() == "tf":
    import tensorflow as tf
elif get_backend() == "pytorch":
    import torch


class NormalDistributionAdapter(ActionAdapter):
    """
    Action adapter for the Normal distribution
    """
    def get_units_and_shape(self):
        # Add moments (2x for each action item).
        units = 2 * self.action_space.flat_dim  # Those two dimensions are the mean and log sd
        if self.action_space.shape == ():
            new_shape = (2,)
        else:
            new_shape = tuple(list(self.action_space.shape[:-1]) + [self.action_space.shape[-1] * 2])
        return units, new_shape

    @graph_fn
    def _graph_fn_get_parameters_log_probs(self, last_nn_layer_output):
        parameters = None
        log_probs = None

        if get_backend() == "tf":
            mean, log_sd = tf.split(last_nn_layer_output, num_or_size_splits=2, axis=-1)
            log_sd = tf.clip_by_value(log_sd, log(SMALL_NUMBER), -log(SMALL_NUMBER))

            # Turn log sd into sd to ascertain always positive stddev values.
            sd = tf.exp(log_sd)
            log_mean = tf.log(mean)

            mean._batch_rank = 0
            sd._batch_rank = 0
            log_mean._batch_rank = 0
            log_sd._batch_rank = 0

            parameters = DataOpTuple([mean, sd])
            log_probs = DataOpTuple([log_mean, log_sd])

        elif get_backend() == "pytorch":
            # Continuous actions.
            mean, log_sd = torch.split(
                last_nn_layer_output, split_size_or_sections=int(parameters.shape[0] / 2), dim=-1
            )
            log_sd = torch.clamp(log_sd, min=log(SMALL_NUMBER), max=-log(SMALL_NUMBER))

            # Turn log sd into sd.
            sd = torch.exp(log_sd)
            log_mean = torch.log(mean)

            parameters = DataOpTuple([mean, sd])
            log_probs = DataOpTuple([log_mean, log_sd])

        return parameters, log_probs
