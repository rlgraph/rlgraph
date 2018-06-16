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
import numpy as np
from six.moves import xrange as range_

from yarl import get_backend, SMALL_NUMBER
from yarl.components import Component
from yarl.components.layers import DenseLayer, DuelingLayer
from yarl.spaces import Space, IntBox, ContainerSpace, sanity_check_space

if get_backend() == "tf":
    import tensorflow as tf


class ActionAdapter(Component):
    """
    A Component that cleans up a neural network's output and gets it ready for parameterizing a Distribution Component.
    Cleanup includes:
    - Sending the NN output through a Dense layer whose number of units matches the flattened action space.
    - Reshaping (for the unflattened action Space).
    - Depending on options: Adding a DuelingLayer.
    - Translating the outputs into probabilities and logits.

    API:
    ins:
        nn_output (SingleDataOp): The raw neural net output to be cleaned up for further processing in a
            Distribution.
    outs:
        action_layer_output (SingleDataOp): The reshaped NN output (e.g. q-values) before being translated into
            Distribution parameters (e.g. via Softmax).
        logits (SingleDataOp): log(action_layer_output).
        parameters (SingleDataOp): The final results of translating the raw NN-output into Distribution-readable
            parameters.
    """
    def __init__(self, biases_spec=False, add_dueling_layer=False, scope="action-adapter", **kwargs):
        """
        Args:
            biases_spec (any): An optional biases_spec to create a biases YARL Initializer that will be used to
                initialize the biases of the ActionLayer.
            #action_output_name (str): The name of the out-Socket to push the raw action output through (before
            #    parameterization for the Distribution).
            #TODO: For now, only IntBoxes -> Categorical are supported. We'll add support for continuous action spaces later
            add_dueling_layer (bool): If True, a DuelingLayer will be inserted after action_layer and reshaping
                steps and the parameterization step.
        """
        super(ActionAdapter, self).__init__(scope=scope, flatten_ops=kwargs.pop("flatten_ops", False), **kwargs)

        self.biases_spec = biases_spec
        self.add_dueling_layer = add_dueling_layer

        self.target_space = None
        self.action_layer = None

        # Define our interface.
        self.define_inputs("nn_output")
        self.define_outputs("action_layer_output", "parameters", "logits")

    def check_input_spaces(self, input_spaces, action_space):
        # Check the input Space.
        in_space = input_spaces["nn_output"]  # type: Space
        sanity_check_space(in_space, non_allowed_types=[ContainerSpace])

        # Check the target (action) Space.
        self.target_space = action_space.with_batch_rank()
        sanity_check_space(self.target_space, must_have_batch_rank=True, allowed_types=[IntBox],
                           must_have_categories=True)

    def create_variables(self, input_spaces, action_space):
        # Create the action layer.
        self.action_layer = DenseLayer(
            units=self.target_space.flat_dim_with_categories + (1 if self.add_dueling_layer is True else 0),
            biases_spec=self.biases_spec if np.isscalar(self.biases_spec) or self.biases_spec is None else
            [log(b) for _ in range_(self.target_space.flat_dim) for b in self.biases_spec]
        )
        self.add_component(self.action_layer)
        self.connect("nn_output", (self.action_layer, "input"))

        # Connect our simple reshape graph_fn after the (dense) action_layer output.
        self.add_graph_fn([(self.action_layer, "output")], "action_layer_output", self._graph_fn_reshape)

        # Add the dueling layer.
        if self.add_dueling_layer:
            dueling_layer = DuelingLayer()
            self.add_component(dueling_layer)
            self.connect("action_layer_output", (dueling_layer, "input"))
            # Connect the dueling layer to our parameters graph_fn.
            self.add_graph_fn([(dueling_layer, "input")], ["logits", "parameters"],
                              self._graph_fn_generate_parameters)
        else:
            # Connect the output of the reshape to the parameters graph_fn.
            self.add_graph_fn("action_layer_output", ["logits", "parameters"],
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
        shape = list(self.target_space.get_shape(with_batch_rank=-1, with_category_rank=True))

        # Add the value function output (1 node) to the last rank.
        if self.add_dueling_layer:
            shape[-1] += 1

        if get_backend() == "tf":
            reshaped = tf.reshape(tensor=action_layer_output, shape=shape)
            return reshaped

    def _graph_fn_generate_parameters(self, reshaped_nn_output):
        """
        Creates properties/parameters and logits from some reshaped output.

        Args:
            reshaped_nn_output (SingleDataOp): The output of some layer that is already reshaped
                according to our action Space.

        Returns:
            tuple:
                "logits" (SingleDataOp): The reshaped action_layer output.
                "parameters" (SingleDataOp): The parameters, ready to be passed to a Distribution object's in-Socket
                    "parameters".
        """
        if get_backend() == "tf":
            # TODO: Missing support for continuous actions.
            parameters = tf.maximum(x=tf.nn.softmax(logits=reshaped_nn_output, axis=-1), y=SMALL_NUMBER)
            # Log probs.
            logits = tf.log(parameters)
            # Convert logits into probabilities and clamp them at SMALL_NUMBER.
            return logits, parameters
