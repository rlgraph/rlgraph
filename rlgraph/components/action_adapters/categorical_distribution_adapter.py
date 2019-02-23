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
    from rlgraph.utils.pytorch_util import SMALL_NUMBER_TORCH


class CategoricalDistributionAdapter(ActionAdapter):
    """Action adapter for the Categorical distribution"""

    def get_units_and_shape(self, add_units=0, units=None):
        if units is None:
            units = add_units + self.action_space.flat_dim_with_categories
        new_shape = self.action_space.get_shape(with_category_rank=True)
        return units, new_shape

    @graph_fn
    def _graph_fn_get_parameters_log_probs(self, logits):
        parameters = None
        log_probs = None

        if get_backend() == "tf":
            parameters = tf.maximum(x=tf.nn.softmax(logits=logits, axis=-1), y=SMALL_NUMBER)
            parameters._batch_rank = 0
            # Log probs.
            log_probs = tf.log(x=parameters)
            log_probs._batch_rank = 0

        elif get_backend() == "pytorch":
            softmax_logits = torch.softmax(logits, dim=-1)
            parameters = torch.max(softmax_logits, SMALL_NUMBER_TORCH)
            # Log probs.
            log_probs = torch.log(parameters)

        return parameters, log_probs
