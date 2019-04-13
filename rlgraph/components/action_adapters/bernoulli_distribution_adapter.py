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

from rlgraph import get_backend
from rlgraph.components.action_adapters import ActionAdapter
from rlgraph.utils.decorators import graph_fn

if get_backend() == "tf":
    import tensorflow as tf
elif get_backend() == "pytorch":
    import torch


class BernoulliDistributionAdapter(ActionAdapter):
    """
    Action adapter for the Bernoulli distribution.
    """
    def get_units_and_shape(self):
        units = self.action_space.flat_dim
        new_shape = self.action_space.get_shape()
        return units, new_shape

    @graph_fn
    def _graph_fn_get_parameters_from_adapter_outputs(self, adapter_outputs):
        parameters = None
        log_probs = None

        if get_backend() == "tf":
            parameters = tf.nn.sigmoid(adapter_outputs)
            parameters._batch_rank = 0
            # Log probs.
            log_probs = tf.log(x=parameters)
            log_probs._batch_rank = 0

        elif get_backend() == "pytorch":
            parameters = torch.sigmoid(adapter_outputs)
            # Log probs.
            log_probs = torch.log(parameters)

        return parameters, log_probs
