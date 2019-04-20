# Copyright 2018/2019 The Rlgraph Authors, All Rights Reserved.
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
from rlgraph.components.neural_networks.neural_network import NeuralNetwork
from rlgraph.components.policies.policy import Policy
from rlgraph.utils.decorators import rlgraph_api
from rlgraph.utils.rlgraph_errors import RLGraphObsoletedError


class SharedValueFunctionPolicy(Policy):
    def __init__(self, network_spec, value_weights_spec=None, value_biases_spec=None, value_activation=None,
                 value_fold_time_rank=False, value_unfold_time_rank=False,
                 scope="shared-value-function-policy", **kwargs):
        super(SharedValueFunctionPolicy, self).__init__(network_spec, scope=scope, **kwargs)

        # Create the extra value dense layer with 1 node.
        self.value_unfold_time_rank = value_unfold_time_rank
        self.value_network = NeuralNetwork(DenseLayer(
            units=1,
            activation=value_activation,
            weights_spec=value_weights_spec,
            biases_spec=value_biases_spec,
        ), fold_time_rank=value_fold_time_rank, unfold_time_rank=value_unfold_time_rank,
            scope="value-function-node")

        self.add_components(self.value_network)

    @rlgraph_api
    def get_state_values(self, nn_inputs):  # , internal_states=None
        """
        Returns the state value node's output.

        Args:
            nn_inputs (any): The input to our neural network.
            #internal_states (Optional[any]): The initial internal states going into an RNN-based neural network.

        Returns:
            Dict:
                state_values: The single (but batched) value function node output.
        """
        nn_outputs = self.get_nn_outputs(nn_inputs)
        #if self.value_unfold_time_rank is True:
        #    state_values = self.value_network.call(nn_outputs, nn_inputs)
        #else:
        state_values = self.value_network.call(nn_outputs)

        return dict(state_values=state_values, nn_outputs=nn_outputs)

    @rlgraph_api
    def get_state_values_adapter_outputs_and_parameters(self, nn_inputs):  #, internal_states=None
        """
        Similar to `get_values_logits_probabilities_log_probs`, but also returns in the return dict under key
        `state_value` the output of our state-value function node.

        Args:
            nn_inputs (any): The input to our neural network.
            #internal_states (Optional[any]): The initial internal states going into an RNN-based neural network.

        Returns:
            Dict:
                nn_outputs: The raw NN outputs.
                state_values: The single (but batched) value function node output.
                adapter_outputs: The (reshaped) logits from the ActionAdapter.
                parameters: The parameters for the distribution (gained from the softmaxed logits or interpreting
                    logits as mean and stddev for a normal distribution).
                log_probs: The log(probabilities) values.
        """
        nn_outputs = self.get_nn_outputs(nn_inputs)
        adapter_outputs, parameters, log_probs = self._graph_fn_get_adapter_outputs_and_parameters(nn_outputs)
        #if self.value_unfold_time_rank is True:
        #    state_values = self.value_network.call(nn_outputs, nn_inputs)
        #else:
        state_values = self.value_network.call(nn_outputs)

        return dict(
            nn_outputs=nn_outputs, state_values=state_values, adapter_outputs=adapter_outputs,
            parameters=parameters, log_probs=log_probs
        )

    def get_state_values_logits_probabilities_log_probs(self, nn_input, internal_states=None):
        raise RLGraphObsoletedError(
            "API-method", "get_state_values_logits_probabilities_log_probs",
            "get_state_values_adapter_outputs_and_parameters"
        )

    def get_state_values_logits_parameters_log_probs(self, nn_input, internal_states=None):
        raise RLGraphObsoletedError(
            "API-method", "get_state_values_logits_parameters_log_probs",
            "get_state_values_adapter_outputs_and_parameters"
        )
