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

from rlgraph.utils import RLGraphError
from rlgraph.components.action_adapters.action_adapter import ActionAdapter
from rlgraph.components.layers.nn.dueling_layer import DuelingLayer


class OBSOLETEDuelingActionAdapter(ActionAdapter):
    """
    An ActionAdapter that adds a DuelingLayer to the output of the base ActionAdapter.

    API:
        get_dueling_output(nn_output) (Tuple[SingleDataOp x 3]): The state-value, advantage-values
            (reshaped) and q-values (reshaped) after passing action_layer_output through the dueling layer.
    """
    def __init__(self, scope="dueling-action-adapter", **kwargs):
        raise RLGraphError("OBSOLETEDuelingActionAdapter has been obsoleted! Use DuelingActionAdapter instead.")

        # Change the number of units in the action layer (+1 for the extra Value function node).
        super(OBSOLETEDuelingActionAdapter, self).__init__(add_units=1, scope=scope, **kwargs)

        # Add the extra DuelingLayer.
        self.dueling_layer = DuelingLayer()
        self.add_components(self.dueling_layer)

    def get_logits_parameters_log_probs(self, nn_output):
        """
        Override get_logits_parameters_log_probs API-method to use (A minus V) Q-values, instead of raw logits from
        network.
        """
        _, _, q_values = self.call(self.get_dueling_output, nn_output)
        return (q_values,) + tuple(self.call(self._graph_fn_get_parameters_log_probs, q_values))

    def get_dueling_output(self, nn_output):
        """
        API-method. Returns separated V, A, and Q-values from the DuelingLayer.

        Args:
            nn_output (DataOpRecord): The NN output of the preceding neural network.

        Returns:
            tuple (3x DataOpRecord):
                - The single state value (V).
                - The advantage values for the different actions.
                - The Q-values for the different actions (calculated as Q=V+A-mean(A), where V and mean(A) are
                broadcast to match A's shape)
        """
        action_layer_output = self.call(self.action_layer.apply, nn_output)
        state_value, advantage_values, q_values = self.call(self.dueling_layer.apply, action_layer_output)
        return state_value, advantage_values, q_values
