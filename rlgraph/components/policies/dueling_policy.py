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

from rlgraph import get_backend
from rlgraph.components.common.softmax import Softmax
from rlgraph.components.layers.nn.dense_layer import DenseLayer
from rlgraph.components.policies.policy import Policy
from rlgraph.utils.decorators import rlgraph_api, graph_fn
from rlgraph.utils.rlgraph_errors import RLGraphObsoletedError
from rlgraph.utils.util import get_rank

if get_backend() == "tf":
    import tensorflow as tf
elif get_backend() == "pytorch":
    import torch


class DuelingPolicy(Policy):
    def __init__(self, network_spec, units_state_value_stream,
                 weights_spec_state_value_stream=None, biases_spec_state_value_stream=None,
                 activation_state_value_stream="relu", scope="dueling-policy", **kwargs):
        super(DuelingPolicy, self).__init__(network_spec, scope=scope, **kwargs)

        self.action_space_flattened = self.action_space.flatten()

        # The state-value stream.
        self.units_state_value_stream = units_state_value_stream
        self.weights_spec_state_value_stream = weights_spec_state_value_stream
        self.biases_spec_state_value_stream = biases_spec_state_value_stream
        self.activation_state_value_stream = activation_state_value_stream

        # Our softmax component to produce probabilities.
        self.softmax = Softmax()

        # Create all state value extra Layers.
        # TODO: Make this a NN-spec as well (right now it's one layer fixed plus the final value node).
        self.dense_layer_state_value_stream = DenseLayer(
            units=self.units_state_value_stream, weights_spec=self.weights_spec_state_value_stream,
            biases_spec=self.biases_spec_state_value_stream,
            activation=self.activation_state_value_stream,
            scope="dense-layer-state-value-stream"
        )
        self.state_value_node = DenseLayer(
            units=1,
            activation="linear",
            scope="state-value-node"
        )

        self.add_components(self.dense_layer_state_value_stream, self.state_value_node)

    @rlgraph_api
    def get_state_values(self, nn_inputs):  #, internal_states=None):
        """
        Returns the state value node's output passing some nn-input through the policy and the state-value
        stream.

        Args:
            nn_inputs (any): The input to our neural network.
            #internal_states (Optional[any]): The initial internal states going into an RNN-based neural network.

        Returns:
            Dict:
                state_values: The single (but batched) value function node output.
        """
        nn_outputs = self.get_nn_outputs(nn_inputs)
        state_values_tmp = self.dense_layer_state_value_stream.call(nn_outputs)
        state_values = self.state_value_node.call(state_values_tmp)

        return dict(state_values=state_values, nn_outputs=nn_outputs)

    @rlgraph_api
    def get_state_values_adapter_outputs_and_parameters(self, nn_inputs):
        """
        Similar to `get_values_logits_probabilities_log_probs`, but also returns in the return dict under key
        `state_value` the output of our state-value function node.

        Args:
            nn_inputs (any): The input to our neural network.
            #internal_states (Optional[any]): The initial internal states going into an RNN-based neural network.

        Returns:
            Dict:
                state_values: The single (but batched) value function node output.
                action_adapter_outputs: The (reshaped) logits from the ActionAdapter.
                parameters: The parameters for the distribution (gained from the softmaxed logits or interpreting
                    logits as mean and stddev for a normal distribution).
                log_probs: The log(probabilities) values.
                last_internal_states: The last internal states (if network is RNN-based).
        """
        nn_outputs = self.get_nn_outputs(nn_inputs)
        advantages, _, _ = self._graph_fn_get_adapter_outputs_and_parameters(nn_outputs)
        state_values_tmp = self.dense_layer_state_value_stream.call(nn_outputs)
        state_values = self.state_value_node.call(state_values_tmp)

        q_values = self._graph_fn_calculate_q_values(state_values, advantages)

        parameters, log_probs = self._graph_fn_get_parameters_from_q_values(q_values)

        return dict(
            nn_outputs=nn_outputs, adapter_outputs=q_values, state_values=state_values,
            parameters=parameters, log_probs=log_probs,
            advantages=advantages, q_values=q_values
        )

    @rlgraph_api
    def get_adapter_outputs(self, nn_inputs):
        """
        Args:
            nn_inputs (any): The input to our neural network.

        Returns:
            Dict:
                nn_outputs: The raw NN outputs.
                adapter_outputs: The q-values after adding advantages to state values (and subtracting the
                    mean advantage).
                advantages:
                q_values:
        """
        nn_outputs = self.get_nn_outputs(nn_inputs)
        advantages, _, _ = self._graph_fn_get_adapter_outputs_and_parameters(nn_outputs)
        state_values_tmp = self.dense_layer_state_value_stream.call(nn_outputs)
        state_values = self.state_value_node.call(state_values_tmp)

        q_values = self._graph_fn_calculate_q_values(state_values, advantages)

        return dict(
            nn_outputs=nn_outputs,
            adapter_outputs=q_values,
            advantages=advantages,
            q_values=q_values
        )

    @rlgraph_api
    def get_adapter_outputs_and_parameters(self, nn_inputs):
        """
        Args:
            nn_inputs (any): The input to our neural network.
            #internal_states (Optional[any]): The initial internal states going into an RNN-based neural network.

        Returns:
            Dict:
                nn_outputs: The raw NN outputs.
                adapter_outputs: The q-values after adding advantages to state values (and subtracting the
                    mean advantage).
                parameters: The parameters for the distribution (gained from the softmaxed logits or interpreting
                    logits as mean and stddev for a normal distribution).
                log_probs: The log(probabilities) values iff we have a discrete action space.
        """
        out = self.get_state_values_adapter_outputs_and_parameters(nn_inputs)
        return dict(
            nn_outputs=out["nn_outputs"],
            adapter_outputs=out["adapter_outputs"],
            parameters=out["parameters"],
            log_probs=out["log_probs"]
        )

    @graph_fn(flatten_ops=True, split_ops=True)
    def _graph_fn_calculate_q_values(self, state_value, advantage_values):
        """
        Args:
            state_value (SingleDataOp): The single node state-value output.
            advantage_values (SingleDataOp): The already reshaped advantage-values.

        Returns:
            SingleDataOp: The calculated, reshaped Q values (for each composite action) based on:
                Q = V + [A - mean(A)]
        """
        # Use the very first node as value function output.
        # Use all following nodes as advantage function output.
        if get_backend() == "tf":
            # Calculate the q-values according to [1] and return.
            mean_advantages = tf.reduce_mean(input_tensor=advantage_values, axis=-1, keepdims=True)

            # Make sure we broadcast the state_value correctly for the upcoming q_value calculation.
            state_value_expanded = state_value
            for _ in range(get_rank(advantage_values) - 2):
                state_value_expanded = tf.expand_dims(state_value_expanded, axis=1)
            q_values = state_value_expanded + advantage_values - mean_advantages

            # q-values
            return q_values

        elif get_backend() == "pytorch":
            mean_advantages = torch.mean(advantage_values, dim=-1, keepdim=True)

            # Make sure we broadcast the state_value correctly for the upcoming q_value calculation.
            state_value_expanded = state_value
            for _ in range(get_rank(advantage_values) - 2):
                state_value_expanded = torch.unsqueeze(state_value_expanded, dim=1)
            q_values = state_value_expanded + advantage_values - mean_advantages

            # q-values
            return q_values

    @graph_fn(flatten_ops=True, split_ops=True, add_auto_key_as_first_param=True)
    def _graph_fn_get_parameters_from_q_values(self, key, q_values):
        """
        """
        out = self.action_adapters[key].get_parameters_from_adapter_outputs(q_values)
        return out["parameters"], out["log_probs"]

    def get_state_values_logits_probabilities_log_probs(self, nn_input, internal_states=None):
        raise RLGraphObsoletedError(
            "API method", "get_state_values_logits_probabilities_log_probs",
            "get_state_values_adpater_outputs_and_parameters"
        )

    def get_logits_probabilities_log_probs(self, nn_input, internal_states=None):
        raise RLGraphObsoletedError(
            "API method", "get_logits_probabilities_log_probs",
            "get_adapter_outputs_and_parameters"
        )
