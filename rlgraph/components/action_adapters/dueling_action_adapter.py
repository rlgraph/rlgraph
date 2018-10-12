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

import math

from rlgraph import get_backend
from rlgraph.components.action_adapters.action_adapter import ActionAdapter
from rlgraph.components.layers.nn.dense_layer import DenseLayer
from rlgraph.spaces import IntBox, FloatBox
from rlgraph.utils.decorators import rlgraph_api, graph_fn
from rlgraph.utils.util import SMALL_NUMBER, get_rank
from rlgraph.utils.ops import DataOpTuple


if get_backend() == "tf":
    import tensorflow as tf
elif get_backend() == "pytorch":
    import torch


class DuelingActionAdapter(ActionAdapter):
    """
    An ActionAdapter that adds a dueling Q calculation to the flattened output of a neural network.

    API:
        get_dueling_output(nn_output) (Tuple[SingleDataOp x 3]): The state-value, advantage-values
            (reshaped) and q-values (reshaped) after passing action_layer_output through the dueling layer.
    """
    def __init__(self, units_state_value_stream, units_advantage_stream,
                 weights_spec_state_value_stream=None, biases_spec_state_value_stream=None,
                 activation_state_value_stream="relu",
                 weights_spec_advantage_stream=None, biases_spec_advantage_stream=None,
                 activation_advantage_stream="relu",
                 scope="dueling-action-adapter", **kwargs):
        # TODO: change add_units=-1 once we have a true base class for action-adapters.
        super(DuelingActionAdapter, self).__init__(add_units=0, scope=scope, **kwargs)

        # The state-value stream.
        self.units_state_value_stream = units_state_value_stream
        self.weights_spec_state_value_stream = weights_spec_state_value_stream
        self.biases_spec_state_value_stream = biases_spec_state_value_stream
        self.activation_state_value_stream = activation_state_value_stream

        # The advantage stream.
        self.units_advantage_stream = units_advantage_stream
        self.weights_spec_advantage_stream = weights_spec_advantage_stream
        self.biases_spec_advantage_stream = biases_spec_advantage_stream
        self.activation_advantage_stream = activation_advantage_stream

        # Create all 4 extra DenseLayers.
        self.dense_layer_state_value_stream = DenseLayer(
            units=self.units_state_value_stream, weights_spec=self.weights_spec_state_value_stream,
            biases_spec=self.biases_spec_state_value_stream,
            activation=self.activation_state_value_stream,
            scope="dense-layer-state-value-stream"
        )
        self.dense_layer_advantage_stream = DenseLayer(
            units=self.units_state_value_stream, weights_spec=self.weights_spec_state_value_stream,
            biases_spec=self.biases_spec_state_value_stream,
            activation=self.activation_state_value_stream,
            scope="dense-layer-advantage-stream"
        )
        self.state_value_node = DenseLayer(
            units=1,
            activation="linear",
            scope="state-value-node"
        )
        # self.action_layer is our advantage layer

        self.add_components(
            self.dense_layer_state_value_stream, self.dense_layer_advantage_stream, self.state_value_node
        )

    @rlgraph_api
    def get_action_layer_output(self, nn_output):
        """
        Args:
            nn_output (DataOpRecord): The NN output of the preceding neural network.

        Returns:
            tuple:
                DataOpRecord: The output of the state-value stream (a DenseLayer) after passing `nn_output` through it.

                DataOpRecord: The output of the advantage-value stream (a DenseLayer) after passing `nn_output` through
                    it. Note: These will be flat advantage nodes that have not been reshaped yet according to the
                    action_space.
        """
        output_state_value_dense = self.dense_layer_state_value_stream.apply(nn_output)
        output_advantage_dense = self.dense_layer_advantage_stream.apply(nn_output)
        state_value_node = self.state_value_node.apply(output_state_value_dense)
        advantage_nodes = self.action_layer.apply(output_advantage_dense)
        return dict(state_value_node=state_value_node, output=advantage_nodes)

    @rlgraph_api
    def get_logits_probabilities_log_probs(self, nn_output):
        """
        Args:
            nn_output (DataOpRecord): The NN output of the preceding neural network.

        Returns:
            tuple (4x DataOpRecord):
                - The single state value node output.
                - The (already reshaped) q-values (the logits).
                - The probabilities obtained by softmaxing the q-values.
                - The log-probs.
        """
        out = self.get_action_layer_output(nn_output)
        advantage_values_reshaped = self.reshape.apply(out["output"])
        q_values = self._graph_fn_calculate_q_values(out["state_value_node"], advantage_values_reshaped)
        probabilities, log_probs = self._graph_fn_get_probabilities_log_probs(q_values)
        return dict(state_values=out["state_value_node"],
                    logits=q_values, probabilities=probabilities, log_probs=log_probs)

    @graph_fn
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
            ## Separate out the single state-value node.
            #state_value, advantages = tf.split(
            #    value=inputs, num_or_size_splits=(1, self.num_advantage_values), axis=-1
            #)
            # Now we have to reshape the advantages according to our action space.
            #shape = list(self.target_space.get_shape(with_batch_rank=-1, with_category_rank=True))
            #advantages = tf.reshape(tensor=advantage_values, shape=shape)
            # Calculate the q-values according to [1] and return.
            mean_advantages = tf.reduce_mean(input_tensor=advantage_values, axis=-1, keepdims=True)

            # Make sure we broadcast the state_value correctly for the upcoming q_value calculation.
            state_value_expanded = state_value
            for _ in range(get_rank(advantage_values) - 2):
                state_value_expanded = tf.expand_dims(state_value_expanded, axis=1)
            q_values = state_value_expanded + advantage_values - mean_advantages

            ## state-value, advantages, q_values
            # q-values
            return q_values
            #tf.squeeze(state_value, axis=-1), advantages,
        elif get_backend() == "pytorch":
            mean_advantages = torch.mean(advantage_values, dim=-1, keepdim=True)

            # Make sure we broadcast the state_value correctly for the upcoming q_value calculation.
            state_value_expanded = state_value
            for _ in range(get_rank(advantage_values) - 2):
                state_value_expanded = torch.unsqueeze(state_value_expanded, dim=1)
            q_values = state_value_expanded + advantage_values - mean_advantages

            ## state-value, advantages, q_values
            # q-values
            return q_values

    # TODO: Use a SoftMax Component instead (uses the same code as the one below).
    @graph_fn
    def _graph_fn_get_probabilities_log_probs(self, logits):
        """
        Creates properties/parameters and log-probs from some reshaped output.

        Args:
            logits (SingleDataOp): The output of some layer that is already reshaped
                according to our action Space.

        Returns:
            tuple (2x SingleDataOp):
                parameters (DataOp): The parameters, ready to be passed to a Distribution object's
                    get_distribution API-method (usually some probabilities or loc/scale pairs).

                log_probs (DataOp): Simply the log(parameters).
        """
        if get_backend() == "tf":
            if isinstance(self.action_space, IntBox):
                # Discrete actions.
                parameters = tf.maximum(x=tf.nn.softmax(logits=logits, axis=-1), y=SMALL_NUMBER)
                # Log probs.
                log_probs = tf.log(x=parameters)
            elif isinstance(self.action_space, FloatBox):
                # Continuous actions.
                mean, log_sd = tf.split(value=logits, num_or_size_splits=2, axis=1)
                # Remove moments rank.
                mean = tf.squeeze(input=mean, axis=1)
                log_sd = tf.squeeze(input=log_sd, axis=1)

                # Clip log_sd. log(SMALL_NUMBER) is negative.
                log_sd = tf.clip_by_value(
                    t=log_sd, clip_value_min=math.log(SMALL_NUMBER), clip_value_max=-math.log(SMALL_NUMBER)
                )

                # Turn log sd into sd.
                sd = tf.exp(x=log_sd)

                parameters = DataOpTuple(mean, sd)
                log_probs = DataOpTuple(tf.log(x=mean), log_sd)
            else:
                raise NotImplementedError
            return parameters, log_probs

        elif get_backend() == "pytorch":
            if isinstance(self.action_space, IntBox):
                # Discrete actions.
                parameters = torch.max(torch.softmax(logits, dim=-1), torch.tensor(SMALL_NUMBER))
                # Log probs.
                log_probs = torch.log(parameters)
            elif isinstance(self.action_space, FloatBox):
                # Continuous actions.
                mean, log_sd = torch.split(logits, split_size_or_sections=2, dim=1)
                # Remove moments rank.
                mean = torch.squeeze(mean, dim=1)
                log_sd = torch.squeeze(log_sd, dim=1)

                # Clip log_sd. log(SMALL_NUMBER) is negative.
                log_sd = torch.clamp(
                    log_sd, min=math.log(SMALL_NUMBER), max=-math.log(SMALL_NUMBER)
                )

                # Turn log sd into sd.
                sd = torch.exp(log_sd)

                parameters = DataOpTuple(mean, sd)
                log_probs = DataOpTuple(torch.log(mean), log_sd)
            else:
                raise NotImplementedError

            return parameters, log_probs
