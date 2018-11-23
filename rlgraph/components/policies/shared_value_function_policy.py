# Copyright 2018 The Rlgraph Authors, All Rights Reserved.
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

from rlgraph.components.layers.nn.dense_layer import DenseLayer
from rlgraph.components.policies.policy import Policy
from rlgraph.utils.decorators import rlgraph_api


class SharedValueFunctionPolicy(Policy):
    def __init__(self, network_spec, value_weights_spec=None, value_biases_spec=None, value_activation=None,
                 scope="shared-value-function-policy", **kwargs):
        super(SharedValueFunctionPolicy, self).__init__(network_spec, scope=scope, **kwargs)

        # Create the extra value dense layer with 1 node.
        self.value_layer = DenseLayer(
            units=1,
            activation=value_activation,
            weights_spec=value_weights_spec,
            biases_spec=value_biases_spec,
            scope="value-function-node"
        )

        self.add_components(self.value_layer)

    @rlgraph_api
    def get_state_values(self, nn_input, internal_states=None):
        """
        Returns the state value node's output.

        Args:
            nn_input (any): The input to our neural network.
            internal_states (Optional[any]): The initial internal states going into an RNN-based neural network.

        Returns:
            Dict:
                state_values: The single (but batched) value function node output.
        """
        nn_output = self.get_nn_output(nn_input, internal_states)
        state_values = self.value_layer.apply(nn_output["output"])

        if self.batch_apply is True:
            state_values = self.unfolder.apply(state_values, nn_input)

        return dict(state_values=state_values, last_internal_states=nn_output.get("last_internal_states"))


    @rlgraph_api
    def get_state_values_logits_probabilities_log_probs(self, nn_input, internal_states=None):
        """
        Similar to `get_values_logits_probabilities_log_probs`, but also returns in the return dict under key
        `state_value` the output of our state-value function node.

        Args:
            nn_input (any): The input to our neural network.
            internal_states (Optional[any]): The initial internal states going into an RNN-based neural network.

        Returns:
            Dict:
                state_values: The single (but batched) value function node output.
                logits: The (reshaped) logits from the ActionAdapter.
                probabilities: The probabilities gained from the softmaxed logits.
                log_probs: The log(probabilities) values.
                last_internal_states: The last internal states (if network is RNN-based).
        """
        nn_output = self.get_nn_output(nn_input, internal_states)
        logits, probabilities, log_probs = self._graph_fn_get_action_adapter_logits_probabilities_log_probs(
            nn_output["output"]
        )
        state_values = self.value_layer.apply(nn_output["output"])

        if self.batch_apply is True:
            state_values = self.unfolder.apply(state_values, nn_input)
            logits = self.unfolder.apply(logits, nn_input)
            probabilities = self.unfolder.apply(probabilities, nn_input)
            log_probs = self.unfolder.apply(log_probs, nn_input)

        return dict(state_values=state_values, logits=logits, probabilities=probabilities, log_probs=log_probs,
                    last_internal_states=nn_output.get("last_internal_states"))
