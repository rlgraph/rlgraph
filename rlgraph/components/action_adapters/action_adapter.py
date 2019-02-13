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

from math import log

from rlgraph import get_backend
from rlgraph.components.neural_networks.neural_network import NeuralNetwork
from rlgraph.components.layers.nn.dense_layer import DenseLayer
from rlgraph.components.layers.preprocessing.reshape import ReShape
from rlgraph.spaces import Space, IntBox, FloatBox, ContainerSpace
from rlgraph.spaces.space_utils import sanity_check_space
from rlgraph.utils.decorators import graph_fn, rlgraph_api
from rlgraph.utils.ops import DataOpTuple
from rlgraph.utils.rlgraph_errors import RLGraphObsoletedError
from rlgraph.utils.util import SMALL_NUMBER

if get_backend() == "tf":
    import tensorflow as tf
elif get_backend() == "pytorch":
    import torch
    from rlgraph.utils.pytorch_util import SMALL_NUMBER_TORCH


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
    def __init__(self, action_space=None, final_shape=None, weights_spec=None, biases_spec=None, activation=None,
                 pre_network_spec=None, scope="action-adapter", **kwargs):
        """
        Args:
            action_space (Optional[Space]): The action Space within which this Component will create actions.
                NOTE: Exactly one of `action_space` of `final_shape` must be provided.

            final_shape (Optional[Tuple[int]): An optional final output shape (in case action_space is not provided).
                If None, will calculate the shape automatically from the given `action_space`.
                NOTE: Exactly one of `action_space` of `final_shape` must be provided.

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
        self.action_space = None
        if action_space is not None:
            self.action_space = action_space.with_batch_rank()
            assert not isinstance(self.action_space, ContainerSpace),\
                "ERROR: ActionAdapter cannot handle ContainerSpaces!"

        self.final_shape = final_shape

        assert (self.action_space is None) != (self.final_shape is None),\
            "ERROR: Exactly one of `action_space` or `final_shape` must be provided!"

        if self.final_shape is None:
            # Calculate the number of nodes in the action layer (DenseLayer object) depending on our `self.action_space`
            if isinstance(self.action_space, IntBox):
                self.final_shape = self.action_space.get_shape(with_category_rank=True)
            else:
                # Two slots for each action-component (mean and log sd).
                if self.action_space.shape == ():
                    self.final_shape = (2,)
                else:
                    self.final_shape = tuple(list(self.action_space.shape[:-1]) + [self.action_space.shape[-1] * 2])
        else:
            assert isinstance(self.final_shape, (tuple, list)), "ERROR: `final_shape` must be a list or a tuple!"

        # Use `self.final_shape` for number of nodes calculation.
        units = 1
        for s in self.final_shape:
            units *= s

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

    def check_input_spaces(self, input_spaces, action_space=None):
        # Check the input Space.
        last_nn_layer_space = input_spaces["nn_output"]  # type: Space
        sanity_check_space(last_nn_layer_space, non_allowed_types=[ContainerSpace])

        # Check the action Space.
        #sanity_check_space(self.action_space, must_have_batch_rank=True)
        # IntBoxes must have categories.
        if isinstance(self.action_space, IntBox):
            sanity_check_space(self.action_space, must_have_categories=True)

    @rlgraph_api
    def get_logits(self, nn_output, nn_input=None):
        """
        Args:
            nn_output (DataOpRecord): The NN output of the preceding neural network.
            nn_input (DataOpRecord): The NN input of the preceeding neural network.
                Only needed if unfold_time_rank is True.

        Returns:
            SingleDataOp: The logits (raw nn_output, BUT reshaped).
        """
        # If we are unfolding and NOT folding -> pass original input in as well.
        if self.api_methods_options[0].get("unfold_time_rank") and \
                not self.api_methods_options[0].get("fold_time_rank"):
            logits_out = self.apply(nn_output, nn_input)
        else:
            logits_out = self.apply(nn_output)
        return logits_out["output"]

    @rlgraph_api
    def get_logits_parameters_log_probs(self, nn_output, nn_input=None):
        """
        Args:
            nn_output (DataOpRecord): The NN output of the preceding neural network.
            nn_input (DataOpRecord): The NN input  of the preceding neural network (needed for optional time-rank
                folding/unfolding purposes).

        Returns:
            Dict[str,SingleDataOp]:
                - "logits": The raw nn_output, only reshaped according to the action_space.
                - "parameters": The softmaxed(logits) for the discrete case and the mean/std values for the continuous
                    case.
                - "log_probs": log([action probabilities])
        """
        logits = self.get_logits(nn_output, nn_input)
        out = self.get_parameters_log_probs(logits)
        return dict(logits=logits, parameters=out["parameters"], log_probs=out["log_probs"])

    def get_logits_probabilities_log_probs(self, nn_output, nn_input=None):
        raise RLGraphObsoletedError(
            "API method", "get_logits_probabilities_log_probs", "get_logits_parameters_log_probs"
        )

    @rlgraph_api
    def get_parameters_log_probs(self, logits):
        """
        Args:
            logits (SingleDataOp): The output of some layer that is already reshaped
                according to our action Space.
        """
        parameters, log_probs = self._graph_fn_get_parameters_log_probs(logits)
        return dict(parameters=parameters, log_probs=log_probs)

    # TODO: Use a SoftMax Component instead (uses the same code as the one below).
    @graph_fn
    def _graph_fn_get_parameters_log_probs(self, logits):
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
        parameters = None
        log_probs = None

        if get_backend() == "tf":
            # Discrete actions.
            if isinstance(self.action_space, IntBox):
                parameters = tf.maximum(x=tf.nn.softmax(logits=logits, axis=-1), y=SMALL_NUMBER)
                parameters._batch_rank = 0
                # Log probs.
                log_probs = tf.log(x=parameters)
                log_probs._batch_rank = 0

            # Continuous actions.
            elif isinstance(self.action_space, FloatBox):
                # Unbounded -> Normal distribution.
                if self.action_space.unbounded:
                    mean, log_sd = tf.split(logits, num_or_size_splits=2, axis=-1)

                    # Turn log sd into sd to ascertain always positive stddev values.
                    sd = tf.exp(log_sd)
                    log_mean = tf.log(mean)

                    mean._batch_rank = 0
                    sd._batch_rank = 0
                    log_mean._batch_rank = 0
                    log_sd._batch_rank = 0

                    parameters = DataOpTuple([mean, sd])
                    log_probs = DataOpTuple([log_mean, log_sd])

                # Bounded -> Beta distribution.
                else:
                    # Stabilize both alpha and beta (currently together in parameters).
                    parameters = tf.clip_by_value(
                        logits, clip_value_min=log(SMALL_NUMBER), clip_value_max=-log(SMALL_NUMBER)
                    )
                    parameters = tf.log((tf.exp(parameters) + 1.0)) + 1.0
                    alpha, beta = tf.split(parameters, num_or_size_splits=2, axis=-1)
                    alpha._batch_rank = 0
                    beta._batch_rank = 0
                    log_alpha = tf.log(alpha)
                    log_beta = tf.log(beta)
                    log_alpha._batch_rank = 0
                    log_beta._batch_rank = 0

                    parameters = DataOpTuple([alpha, beta])
                    log_probs = DataOpTuple([log_alpha, log_beta])

        elif get_backend() == "pytorch":
            if isinstance(self.action_space, IntBox):
                # Discrete actions.
                softmax_logits = torch.softmax(logits, dim=-1)
                parameters = torch.max(softmax_logits, SMALL_NUMBER_TORCH)
                # Log probs.
                log_probs = torch.log(parameters)
            elif isinstance(self.action_space, FloatBox):
                # Unbounded -> Normal distribution.
                if self.action_space.unbounded:
                    # Continuous actions.
                    mean, log_sd = torch.split(logits, split_size_or_sections=int(parameters.shape[0] / 2), dim=-1)

                    # Turn log sd into sd.
                    sd = torch.exp(log_sd)
                    log_mean =  torch.log(mean)

                    parameters = DataOpTuple([mean, sd])
                    log_probs = DataOpTuple([log_mean, log_sd])

                # Bounded -> Beta distribution.
                else:
                    # Stabilize both alpha and beta (currently together in parameters).
                    parameters = torch.clamp(
                        logits, min=log(SMALL_NUMBER), max=-log(SMALL_NUMBER)
                    )
                    parameters = torch.log((torch.exp(parameters) + 1.0)) + 1.0

                    # Split in the middle.
                    alpha, beta = torch.split(parameters, split_size_or_sections=int(parameters.shape[0] / 2), dim=-1)
                    log_alpha = torch.log(alpha)
                    log_beta = torch.log(beta)

                    parameters = DataOpTuple([alpha, beta])
                    log_probs = DataOpTuple([log_alpha, log_beta])

        return parameters, log_probs
