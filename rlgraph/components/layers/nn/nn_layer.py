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

from rlgraph import get_backend
from rlgraph.components.layers.layer import Layer
from rlgraph.components.layers.nn.activation_functions import get_activation_function
from rlgraph.spaces import FloatBox, IntBox
from rlgraph.spaces.space_utils import sanity_check_space
from rlgraph.utils.decorators import rlgraph_api


class NNLayer(Layer):
    """
    A generic NN-layer object implementing the `call` graph_fn and offering additional activation function support.
    Can be used in the following ways:

    - Thin wrapper around a backend-specific layer object (normal use case):
        Create the backend layer in the `create_variables` method and store it under `self.layer`. Then register
        the backend layer's variables with the RLgraph Component.

    - Custom layer (with custom computation):
        Create necessary variables in `create_variables` (e.g. matrices), then override `_graph_fn_call`, leaving
        `self.layer` as None.

    - Single Activation Function:
        Leave `self.layer` as None and do not override `_graph_fn_call`. It will then only apply the activation
        function.
    """
    def __init__(self, **kwargs):
        # Most NN layers have an activation function (some with parameters e.g. leaky ReLU).
        self.activation = kwargs.pop("activation", None)
        self.activation_params = kwargs.pop("activation_params", [])

        # Activation fn for define-by-run execution.
        self.activation_fn = None

        # The wrapped backend-layer object.
        self.layer = None
        self.in_space_0 = None
        self.time_major = None

        super(NNLayer, self).__init__(scope=kwargs.pop("scope", "nn-layer"), **kwargs)

    def check_input_spaces(self, input_spaces, action_space=None):
        """
        Do some sanity checking on the incoming Space:
        Must not be Container (for now) and must have a batch rank.
        """
        # Make sure all inputs have the same time/batch ranks.
        # TODO also check spaces for pytorch once unified space management
        if get_backend() == "tf":
            if "inputs[0]" in input_spaces:
                self.in_space_0 = input_spaces["inputs[0]"]
                self.time_major = self.in_space_0.time_major
                idx = 0
                while True:
                    key = "inputs[{}]".format(idx)
                    if key not in input_spaces:
                        break
                    sanity_check_space(
                        input_spaces[key], allowed_sub_types=[FloatBox, IntBox], must_have_batch_rank=True
                    )
                    # Make sure all concat inputs have same batch-/time-ranks.
                    assert self.in_space_0.has_batch_rank == input_spaces[key].has_batch_rank and \
                        self.in_space_0.has_time_rank == input_spaces[key].has_time_rank, \
                        "ERROR: Input spaces to '{}' must have same batch-/time-rank structure! " \
                        "0th input is batch-rank={} time-rank={}, but {}st input is batch-rank={} " \
                        "time-rank={}.".format(
                            self.global_scope, self.in_space_0.has_batch_rank, input_spaces[key].has_batch_rank, idx,
                            self.in_space_0.has_time_rank, input_spaces[key].has_time_rank
                        )

                    idx += 1

    @rlgraph_api
    def _graph_fn_call(self, *inputs):
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
                output = activation_function(*inputs)
                # TODO: Move into util function.
                # Add batch-/time-rank flags.
                output._batch_rank = 0 if self.time_major is False else 1
                if self.in_space_0 and self.in_space_0.has_time_rank:
                    output._time_rank = 0 if self.in_space_0.time_major is True else 1
                return output
        # `self.layer` already includes activation function details.
        else:
            if get_backend() == "tf":
                output = self.layer.call(*inputs)
                # Add batch-/time-rank flags.
                output._batch_rank = 0 if self.time_major is False else 1
                if self.in_space_0 and self.in_space_0.has_time_rank:
                    output._time_rank = 0 if self.in_space_0.time_major is True else 1
                return output
            elif get_backend() == "pytorch":
                # Strip empty internal states:
                # Ensure inputs are float tensors.
                input_tensors = []
                for value in inputs:
                    if value is not None and hasattr(value, "float"):
                        input_tensors.append(value.float())
                if not input_tensors:
                    return None

                # Common debug print:
                # print("in net work layer: ", self.name)
                # import torch
                # shapes = []
                # for inp in inputs:
                #     if hasattr(inp, "shape"):
                #         shapes.append(inp.shape)
                #     else:
                #         shapes.append(type(inp))
                # print("input shapes = ", shapes)
                # PyTorch layers are called, not `applied`.
                out = self.layer(*input_tensors)
                # print("layer output shape = ", out.shape)
                if self.activation_fn is None:
                    return out
                else:
                    # Apply activation fn.
                    return self.activation_fn(out)
