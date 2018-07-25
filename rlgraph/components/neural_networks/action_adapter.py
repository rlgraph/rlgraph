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

from math import log

from rlgraph import get_backend, SMALL_NUMBER
from rlgraph.components import Component
from rlgraph.components.layers.nn import DenseLayer, DuelingLayer
from rlgraph.spaces import Space, IntBox, FloatBox, ContainerSpace, sanity_check_space

if get_backend() == "tf":
    import tensorflow as tf


class ActionAdapter(Component):
    """
    A Component that cleans up a neural network's flat output and gets it ready for parameterizing a
    Distribution Component.
    Processing steps include:
    - Sending the raw, flattened NN output through a Dense layer whose number of units matches the flattened
        action space (+1 if `add_dueling_layer` is True).
    - Depending on options:
        - Either: Adding a DuelingLayer (with own reshape).
        - Or: Reshaping (according to the action Space).
    - Translating the reshaped outputs into probabilities and logits.

    API:
        get_action_layer_output(nn_output) (SingleDataOp): The raw, non-reshaped output of the action-layer
            (DenseLayer) after passing through the raw nn_output (from the previous Component).
        get_logits_parameters_log_probs(nn_output) (Tuple[SingleDataOp x 3]):
            1) raw nn_output, BUT reshaped
            2) probabilities (softmaxed (1))
            3) log(probabilities)

        Optional:
            If add_dueling_layer=True:
                get_dueling_output(nn_output) (Tuple[SingleDataOp x 3]): The state-value, advantage-values
                    (reshaped) and q-values (reshaped) after passing action_layer_output through the dueling layer.
    """
    def __init__(self, action_space, weights_spec=None, biases_spec=None, activation=None, add_dueling_layer=False,
                 scope="action-adapter", **kwargs):
        """
        Args:
            action_space (Space): The action Space within which this Component will create actions.
            weights_spec (Optional[]): An optional RLGraph Initializer spec that will be used to initialize the weights
                of `self.action layer`. Default: None (use default initializer).
            biases_spec (any): An optional RLGraph Initializer spec that will be used to initialize the biases
                of `self.action layer`. Default: False (no biases).
            activation (Optional[str]): The activation function to use for `self.action_layer`.
                Default: None (=linear).
            add_dueling_layer (bool): If True, a DuelingLayer will be inserted after action_layer and reshaping
                steps and the parameterization step.
                Default: False.
        """
        super(ActionAdapter, self).__init__(scope=scope, **kwargs)

        self.action_space = action_space.with_batch_rank()
        self.weights_spec = weights_spec
        self.biases_spec = biases_spec
        self.activation = activation

        # Our (dense) action layer representing the flattened action space.
        self.action_layer = None

        # An optional dueling layer after the action_layer.
        self.add_dueling_layer = add_dueling_layer
        self.dueling_layer = None

        # Create the action layer (DenseLayer object) depending on our action Space.
        if isinstance(self.action_space, IntBox):
            units = self.action_space.flat_dim_with_categories
        else:
            units = 2 * self.action_space.flat_dim  # Those two dimensions are the mean and log sd

        self.action_layer = DenseLayer(
            units=units + (1 if self.add_dueling_layer is True else 0),
            activation=self.activation,
            weights_spec=self.weights_spec,
            biases_spec=self.biases_spec,
            scope="action-layer"
        )
        # And connect it to the incoming "nn_output".
        self.add_components(self.action_layer)

        # With dueling layer: Provide dueling_output with state/advantage/q-values.
        if self.add_dueling_layer:
            self.dueling_layer = DuelingLayer()
            self.add_components(self.dueling_layer)

            def get_dueling_output(self_, nn_output):
                action_layer_output = self_.call(self_.action_layer.apply, nn_output)
                state_value, advantage_values, q_values = self_.call(self_.dueling_layer.apply, action_layer_output)
                return state_value, advantage_values, q_values

            self.define_api_method("get_dueling_output", get_dueling_output)

            def get_logits_parameters_log_probs(self_, nn_output):
                _, _, q_values = self_.call(self_.get_dueling_output, nn_output, ok_to_call_own_api=True)
                return self_.call(self_._graph_fn_get_logits_parameters_log_probs, q_values)

            self.define_api_method("get_logits_parameters_log_probs", get_logits_parameters_log_probs)

        # Without dueling layer: Provide raw, reshaped action-layer output.
        else:
            def get_logits_parameters_log_probs(self_, nn_output):
                action_layer_output = self_.call(self_.action_layer.apply, nn_output)
                action_layer_output_reshaped = self_.call(self_._graph_fn_reshape, action_layer_output)
                return self_.call(self_._graph_fn_get_logits_parameters_log_probs, action_layer_output_reshaped)

            self.define_api_method("get_logits_parameters_log_probs", get_logits_parameters_log_probs)

    def check_input_spaces(self, input_spaces, action_space):
        # Check the input Space.
        if "get_logits_parameters_log_probs" in input_spaces:
            last_nn_layer_space = input_spaces["get_logits_parameters_log_probs"][0]  # type: Space
            sanity_check_space(last_nn_layer_space, non_allowed_types=[ContainerSpace])

        # Check the action Space.
        if isinstance(self.action_space, IntBox):
            sanity_check_space(self.action_space, must_have_batch_rank=True, allowed_types=[IntBox],
                               must_have_categories=True)
        else:
            # Fixme: Are there other restraints on continuous action spaces? E.g. no dueling layers?
            sanity_check_space(self.action_space, must_have_batch_rank=True, allowed_types=[FloatBox])

    def get_action_layer_output(self, nn_output):
        return self.call(self.action_layer.apply, nn_output)

    def _graph_fn_reshape(self, action_layer_output):
        """
        Only reshapes some NN output according to our action space.

        Args:
            action_layer_output (SingleDataOp): The output of the action_layer of this Component (last, flattened data
                coming from the NN).

        Returns:
            SingleDataOp: The reshaped action_layer_output.
        """
        # Reshape action_output to action shape.
        if isinstance(self.action_space, IntBox):
            shape = list(self.action_space.get_shape(with_batch_rank=-1, with_category_rank=True))
        elif isinstance(self.action_space, FloatBox):
            shape = [-1, 2] + list(self.action_space.get_shape(with_batch_rank=False))  # Manually add moments rank
        else:
            raise NotImplementedError

        if get_backend() == "tf":
            action_layer_output_reshaped = tf.reshape(tensor=action_layer_output, shape=shape)
            return action_layer_output_reshaped

    def _graph_fn_get_logits_parameters_log_probs(self, logits):
        """
        Creates properties/parameters and logits from some reshaped output.

        Args:
            logits (SingleDataOp): The output of some layer that is already reshaped
                according to our action Space.

        Returns:
            tuple:
                "logits" (SingleDataOp): The log(parameters) values.
                "parameters" (SingleDataOp): The parameters, ready to be passed to a Distribution object's
                    get_distribution API-method (usually some probabilities or loc/scale pairs).
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
                log_sd = tf.clip_by_value(t=log_sd, clip_value_min=log(SMALL_NUMBER), clip_value_max=-log(SMALL_NUMBER))

                # Turn log sd into sd.
                sd = tf.exp(x=log_sd)

                parameters = (mean, sd)
                log_probs = (tf.log(x=mean), log_sd)
            else:
                raise NotImplementedError

            return logits, parameters, log_probs
