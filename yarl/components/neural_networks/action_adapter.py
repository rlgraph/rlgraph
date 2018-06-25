# Copyright 2018 The YARL-Project, All Rights Reserved.
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

from yarl import get_backend, SMALL_NUMBER
from yarl.components import Component
from yarl.components.layers import DenseLayer, DuelingLayer
from yarl.spaces import Space, IntBox, FloatBox, Tuple, ContainerSpace, sanity_check_space

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
    ins:
        nn_output (SingleDataOp): The raw, flattened neural network output to be processed by this ActionAdapter.
    outs:
        action_layer_output (SingleDataOp): The `nn_output` sent through `self.action_layer`.
        Optional:
            If add_dueling_layer=True:
                state_value (SingleDataOp): The state value diverged from the first output node of the previous layer.
                advantage_values (SingleDataOp): The advantage values (already reshaped) for the different actions.
                q_values (SingleDataOp): The Q-values (already reshaped) for the different state-action pairs.
                    Calculated according to the dueling layer logic.
            else:
                action_layer_output_reshaped (SingleDataOp): The action layer output, reshaped according to the action
                    space.
        parameters (SingleDataOp): The final results of translating nn_output into Distribution-readable
            parameters (e.g. probabilities).
        logits (SingleDataOp): Usually just `log(parameters)`.
    """
    def __init__(self, weights_spec=None, biases_spec=None, activation=None, add_dueling_layer=False,
                 scope="action-adapter", **kwargs):
        """
        Args:
            weights_spec (Optional[]): An optional YARL Initializer spec that will be used to initialize the weights
                of `self.action layer`. Default: None (use default initializer).
            biases_spec (any): An optional YARL Initializer spec that will be used to initialize the biases
                of `self.action layer`. Default: False (no biases).
            activation (Optional[str]): The activation function to use for `self.action_layer`.
                Default: None (=linear).
            add_dueling_layer (bool): If True, a DuelingLayer will be inserted after action_layer and reshaping
                steps and the parameterization step.
                Default: False.
        """
        super(ActionAdapter, self).__init__(scope=scope, **kwargs)

        self.weights_spec = weights_spec
        self.biases_spec = biases_spec
        self.activation = activation
        self.target_space = None

        # Our (dense) action layer representing the flattened action space.
        self.action_layer = None

        # An optional dueling layer after the action_layer.
        self.add_dueling_layer = add_dueling_layer
        self.dueling_layer = None

        # Define our interface.
        self.define_inputs("nn_output")
        self.define_outputs("action_layer_output", "parameters", "logits")
        if self.add_dueling_layer is True:
            self.define_outputs("state_value", "advantage_values", "q_values")
        else:
            self.define_outputs("action_layer_output_reshaped")

    def check_input_spaces(self, input_spaces, action_space):
        # Check the input Space.
        in_space = input_spaces["nn_output"]  # type: Space
        sanity_check_space(in_space, non_allowed_types=[ContainerSpace])

        # Check the target (action) Space.
        self.target_space = action_space.with_batch_rank()

        if isinstance(self.target_space, IntBox):
            sanity_check_space(self.target_space, must_have_batch_rank=True, allowed_types=[IntBox],
                               must_have_categories=True)
        else:
            # Fixme: Are there other restraints on continuous action spaces? E.g. no dueling layers?
            sanity_check_space(self.target_space, must_have_batch_rank=True, allowed_types=[FloatBox])

    def create_variables(self, input_spaces, action_space):
        # Create the action layer.
        if isinstance(self.target_space, IntBox):
            units = self.target_space.flat_dim_with_categories
        else:
            units = 2 * self.target_space.flat_dim  # Those two dimensions are the mean and log sd

        self.action_layer = DenseLayer(
            units=units + (1 if self.add_dueling_layer is True else 0),
            activation=self.activation,
            weights_spec=self.weights_spec,
            biases_spec=self.biases_spec,
            scope="action-layer"
        )
        # And connect it to the incoming "nn_output".
        self.add_component(self.action_layer)
        self.connect("nn_output", (self.action_layer, "input"))

        # Expose the action_layer's output via "action_layer_output".
        self.connect((self.action_layer, "output"), "action_layer_output")
        #self.action_layer["output"] > self["action_layer_output"]

        # Add an optional dueling layer.
        if self.add_dueling_layer:
            self.dueling_layer = DuelingLayer()
            self.add_component(self.dueling_layer)
            self.connect("action_layer_output", (self.dueling_layer, "input"))
            self.connect((self.dueling_layer, "state_value"), "state_value")
            self.connect((self.dueling_layer, "advantage_values"), "advantage_values")
            self.connect((self.dueling_layer, "q_values"), "q_values")
            # Create parameters and logits from the q_values of the dueling layer.
            self.add_graph_fn([(self.dueling_layer, "q_values")], ["logits", "parameters"],
                              self._graph_fn_generate_parameters)
        # Without dueling layer.
        else:
            # Connect the output of the action layer to our reshape graph_fn.
            self.add_graph_fn("action_layer_output", "action_layer_output_reshaped",
                              self._graph_fn_reshape)
            # Then to our generate_parameters graph_fn.
            self.add_graph_fn("action_layer_output_reshaped", ["logits", "parameters"],
                              self._graph_fn_generate_parameters)

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
        if isinstance(self.target_space, IntBox):
            shape = list(self.target_space.get_shape(with_batch_rank=-1, with_category_rank=True))
        elif isinstance(self.target_space, FloatBox):
            shape = [-1, 2] + list(self.target_space.get_shape(with_batch_rank=False))  # Manually add moments rank
        else:
            raise NotImplementedError

        if get_backend() == "tf":
            action_layer_output_reshaped = tf.reshape(tensor=action_layer_output, shape=shape)
            return action_layer_output_reshaped

    def _graph_fn_generate_parameters(self, action_layer_output_reshaped):
        """
        Creates properties/parameters and logits from some reshaped output.

        Args:
            action_layer_output_reshaped (SingleDataOp): The output of some layer that is already reshaped
                according to our action Space.

        Returns:
            tuple:
                "parameters" (SingleDataOp): The parameters, ready to be passed to a Distribution object's in-Socket
                    "parameters" (usually some probabilities or loc/scale pairs).
                "logits" (SingleDataOp): The log(parameters) values.
        """
        if get_backend() == "tf":
            if isinstance(self.target_space, IntBox):
                # Discrete actions
                parameters = tf.maximum(x=tf.nn.softmax(logits=action_layer_output_reshaped, axis=-1), y=SMALL_NUMBER)

                # Log probs.
                logits = tf.log(parameters)
            elif isinstance(self.target_space, FloatBox):
                # Continuous actions
                mean, log_sd = tf.split(action_layer_output_reshaped, 2, 1)

                # Remove moments rank
                mean = tf.squeeze(mean, axis=1)
                log_sd = tf.squeeze(log_sd, axis=1)

                # Clip log_sd. log(SMALL_NUMBER) is negative
                log_sd = tf.clip_by_value(t=log_sd, clip_value_min=log(SMALL_NUMBER), clip_value_max=-log(SMALL_NUMBER))

                # Turn log sd into sd
                sd = tf.exp(log_sd)

                logits = (tf.log(mean), log_sd)
                parameters = (mean, sd)
            else:
                raise NotImplementedError

            # Convert logits into probabilities and clamp them at SMALL_NUMBER.
            return logits, parameters
