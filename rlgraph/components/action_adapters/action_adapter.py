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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from rlgraph.components.layers.nn.dense_layer import DenseLayer
from rlgraph.components.layers.preprocessing.reshape import ReShape
from rlgraph.components.neural_networks.neural_network import NeuralNetwork
from rlgraph.spaces import Space, ContainerSpace
from rlgraph.spaces.space_utils import sanity_check_space
from rlgraph.utils.decorators import graph_fn, rlgraph_api
from rlgraph.utils.rlgraph_errors import RLGraphObsoletedError


# TODO: Create a more primitive base class only defining the API-methods.
# Then rename this into `SingleLayerActionAdapter`.
class ActionAdapter(NeuralNetwork):
    """
    A Component that cleans up a neural network's flat output and gets it ready for parameterizing a
    Distribution Component.
    Processing steps include:
    - Sending the raw, flattened NN output through a Dense layer whose number of units matches the flattened
    action space.
    - Reshaping (according to the action Space).
    - Translating the reshaped outputs (logits) into probabilities (by softmaxing) and log-probabilities (log).
    """
    def __init__(self, action_space, weights_spec=None, biases_spec=None, activation=None,
                 pre_network_spec=None, scope="action-adapter", **kwargs):
        """
        Args:
            action_space (Optional[Space]): The action Space within which this Component will create actions.
                NOTE: Exactly one of `action_space` of `final_shape` must be provided.

            #final_shape (Optional[Tuple[int]): An optional final output shape (in case action_space is not provided).
            #    If None, will calculate the shape automatically from the given `action_space`.
            #    NOTE: Exactly one of `action_space` of `final_shape` must be provided.

            weights_spec (Optional[any]): An optional RLGraph Initializer spec that will be used to initialize the
                weights of `self.action layer`. Default: None (use default initializer).

            biases_spec (Optional[any]): An optional RLGraph Initializer spec that will be used to initialize the
                biases of `self.action layer`. Default: None (use default initializer, which is usually 0.0).

            activation (Optional[str]): The activation function to use for `self.action_layer`.
                Default: None (=linear).

            pre_network_spec (Optional[dict,NeuralNetwork]): A spec dict for a neural network coming before the
                last action layer. If None, only the action layer itself is applied.
        """
        # Build the action layer for this adapter based on the given action-space.
        self.action_space = action_space.with_batch_rank()
        assert not isinstance(self.action_space, ContainerSpace),\
            "ERROR: ActionAdapter cannot handle ContainerSpaces!"

        units, self.final_shape = self.get_units_and_shape()
        assert isinstance(units, int) and units > 0, "ERROR: `units` must be int and larger 0!"

        action_layer = DenseLayer(
            units=units,
            activation=activation,
            weights_spec=weights_spec,
            biases_spec=biases_spec,
            scope="action-layer"
        )

        # Do we have a pre-NN?
        self.network = NeuralNetwork.from_spec(pre_network_spec, scope="action-network")  # type: NeuralNetwork
        self.network.add_layer(action_layer)

        # Add the reshape layer to match the action space's shape.
        self.network.add_layer(ReShape(new_shape=self.final_shape))

        super(ActionAdapter, self).__init__(self.network, scope=scope, **kwargs)

    def get_units_and_shape(self):
        """
        Returns the number of units in the layer that will be added and the shape of the output according to the
        action space.

        Returns:
            Tuple:
                int: The number of units for the action layer.
                shape: The final shape for the output space.
        """
        raise NotImplementedError

    def check_input_spaces(self, input_spaces, action_space=None):
        # Check the input Space.
        last_nn_layer_space = input_spaces["inputs[0]"]  # type: Space
        sanity_check_space(last_nn_layer_space, non_allowed_types=[ContainerSpace])
        # Check the action Space.
        #sanity_check_space(self.action_space, must_have_batch_rank=True)

    #@rlgraph_api
    #def get_action_adapter_outputs(self, nn_input, original_nn_input=None):
    #    """
    #    Args:
    #        nn_input (DataOpRecord): The NN output of the preceding neural network.
    #        original_nn_input (DataOpRecord): The NN input of the preceding neural network.
    #            Only needed if unfold_time_rank is True.

    #    Returns:
    #        SingleDataOp: The logits (action layer outputs + reshaped).
    #    """
    #    # If we are unfolding and NOT folding -> pass original input in as well.
    #    if self.api_methods_options[0].get("unfold_time_rank") and \
    #            not self.api_methods_options[0].get("fold_time_rank"):
    #        logits_out = self.call(nn_input, original_nn_input)
    #    else:
    #        logits_out = self.call(nn_input)
    #    return logits_out

    @rlgraph_api
    def get_parameters(self, *inputs):  #, original_nn_input=None):
        """
        Args:
            inputs (DataOpRecord): The NN output(s) of the preceding neural network.
            #original_nn_input (DataOpRecord): The NN input  of the preceding neural network (needed for optional time-rank
            #    folding/unfolding purposes).

        Returns:
            Dict[str,SingleDataOp]:
                - "adapter_outputs": The raw nn_input, only reshaped according to the action_space.
                - "parameters": The softmaxed(logits) for the discrete case and the mean/std values for the continuous
                    case.
                - "log_probs": log([action probabilities]) iff discrete actions.
        """
        #nn_outputs = self.get_action_adapter_outputs(nn_input, original_nn_input)
        adapter_outputs = self.call(*inputs)  #, original_nn_input)
        out = self.get_parameters_from_adapter_outputs(adapter_outputs)
        return dict(adapter_outputs=adapter_outputs, parameters=out["parameters"], log_probs=out["log_probs"])

    @rlgraph_api(must_be_complete=False)
    def get_parameters_from_adapter_outputs(self, adapter_outputs):
        """
        Args:
            adapter_outputs (SingleDataOp): The (action-space reshaped) output of the action adapter's action layer.
        """
        parameters, log_probs = self._graph_fn_get_parameters_from_adapter_outputs(adapter_outputs)
        return dict(parameters=parameters, log_probs=log_probs)

    # TODO: Use a SoftMax Component instead (uses the same code as the one below).
    @graph_fn
    def _graph_fn_get_parameters_from_adapter_outputs(self, adapter_outputs):
        """
        Creates properties/parameters and log-probs from some reshaped output.

        Args:
            adapter_outputs (SingleDataOp): The output of some layer that is already reshaped
                according to our action Space.

        Returns:
            tuple (2x SingleDataOp):
                parameters (DataOp): The parameters, ready to be passed to a Distribution object's
                    get_distribution API-method (usually some probabilities or loc/scale pairs).
                log_probs (DataOp): Simply the log(parameters).
        """
        raise NotImplementedError

    def get_logits(self, nn_input, original_nn_input=None):
        raise RLGraphObsoletedError("API-method", "get_logits", "call")

    def get_logits_parameters_log_probs(self, nn_input, original_nn_input=None):
        raise RLGraphObsoletedError(
            "API method", "get_logits_parameters_log_probs", "get_parameters"
        )

    def get_logits_probabilities_log_probs(self, nn_input, original_nn_input=None):
        raise RLGraphObsoletedError(
            "API method", "get_logits_probabilities_log_probs", "get_parameters"
        )

    def get_parameters_log_probs(self, logits):
        raise RLGraphObsoletedError("API-method", "get_parameters_log_probs", "get_parameters_from_adapter_outputs")
