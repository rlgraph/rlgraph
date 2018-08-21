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

from rlgraph import get_backend
from rlgraph.components.layers.layer import Layer
from rlgraph.components.layers.nn.activation_functions import get_activation_function
from rlgraph.spaces import FloatBox, IntBox
from rlgraph.spaces.space_utils import sanity_check_space


class NNLayer(Layer):
    """
    A generic NN-layer object implementing the `apply` graph_fn and offering additional activation function support.
    Can be used in the following ways:
    - Thin wrapper around a backend-specific layer object (normal use case):
        Create the backend layer in the `create_variables` method and store it under `self.layer`. Then register
        the backend layer's variables with the RLgraph Component.
    - Custom layer (with custom computation):
        Create necessary variables in `create_variables` (e.g. matrices), then override `_graph_fn_apply`, leaving
        `self.layer` as None.
    - Single Activation Function:
        Leave `self.layer` as None and do not override `_graph_fn_apply`. It will then only apply the activation
        function.
    """
    def __init__(self, **kwargs):
        # Most NN layers have an activation function (some with parameters e.g. leaky ReLU).
        self.activation = kwargs.pop("activation", None)
        self.activation_params = kwargs.pop("activation_params", [])

        # The wrapped backend-layer object.
        self.layer = None

        super(NNLayer, self).__init__(scope=kwargs.pop("scope", "nn-layer"), **kwargs)

    def check_input_spaces(self, input_spaces, action_space=None):
        """
        Do some sanity checking on the incoming Space:
        Must not be Container (for now) and must have a batch rank.
        """
        idx = 0
        while True:
            key = "inputs[{}]".format(idx)
            if key not in input_spaces:
                break
            sanity_check_space(input_spaces[key], allowed_types=[FloatBox, IntBox], must_have_batch_rank=True)
            idx += 1

    def _graph_fn_apply(self, *inputs):
        """
        The actual calculation on one or more input Ops.

        Args:
            inputs (SingleDataOp): The single (non-container) input(s) to the layer.

        Returns:
            The output(s) after having pushed input(s) through the layer.
        """
        # `self.layer` is not given: Only apply the activation function.
        if self.layer is None:
            # No activation function.
            if self.activation is None:
                return tuple(inputs)
            # Pass inputs through activation function.
            else:
                activation_function = get_activation_function(self.activation, self.activation_params)
                return activation_function(*inputs)
        # `self.layer` already includes activation function details.
        else:
            if get_backend() == "tf":
                return self.layer.apply(*inputs)
            elif get_backend() == "pytorch":
                # PyTorch layers are called, not applied.
                return self.layer(*inputs)
