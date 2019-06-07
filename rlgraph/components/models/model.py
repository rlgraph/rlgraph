# Copyright 2018/2019 ducandu GmbH. All Rights Reserved.
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

from __future__ import absolute_import, division, print_function

from rlgraph.components.component import Component
from rlgraph.utils.decorators import rlgraph_api


class Model(Component):
    """
    A Model is an abstract Component class that must implement the APIs `predict`, `get_distribution_parameters`, and
    `update`.
    """

    @rlgraph_api
    def predict(self, nn_inputs, deterministic=None):
        """
        Args:
            nn_inputs (any): The input to our neural network.

            deterministic (Optional[bool]): Whether to draw the prediction sample deterministically
                (max likelihood) from the parameterized distribution or not.

        Returns:
            dict:
                - predictions: The final sample from the Distribution (including the entire output of the neural network).
                - nn_outputs: The raw NN outputs
                - adapter_outputs: The NN outputs passed through the action adapter.
                - parameters: The parameters for the distributions.
                - log_likelihood: The log-likelihood (in case of categorical distribution(s)).
        """
        raise NotImplementedError

    @rlgraph_api
    def get_distribution_parameters(self, nn_inputs):
        """
        Args:
            nn_inputs (any): The input to our neural network.

        Returns:
            any: The raw (parameter) output of the DistributionAdapter layer
                (including possibly the last internal states of an RNN-based NN).
        """
        raise NotImplementedError

    @rlgraph_api
    def update(self, nn_inputs, labels, timestep=None):
        """
        Updates the model by doing a forward pass through the predictor, passing its outputs and the
        `loss_function_inputs` through the loss function, then performing the optimizer update.

        Args:
            nn_inputs (any): The inputs to the predictor. Will be passed as one DataOpRecord into the NN.
            labels (any): The corresponding labels for the prediction inputs.
            timestep (Optional[int]): The timestep of the update. Can be used e.g. for decaying learning rates.

        Returns:
            dict with keys:
                - step_op
                - loss
                - loss_per_item
                - parameters: Distribution parameters before the outputs.
        """
        raise NotImplementedError
