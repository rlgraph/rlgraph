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

from rlgraph.components.layers.layer import Layer
from rlgraph.spaces import FloatBox, sanity_check_space


class NNLayer(Layer):
    """
    A generic NN-layer object.
    """
    def __init__(self, **kwargs):
        # Most NN layers have an activation function (some with parameters e.g. leaky ReLU).
        self.activation = kwargs.pop("activation", None)
        self.activation_params = kwargs.pop("activation_params", [])

        # The wrapped backend-layer object.
        self.layer = None

        super(NNLayer, self).__init__(scope=kwargs.pop("scope", "nn-layer"), **kwargs)

    def check_input_spaces(self, input_spaces, action_space):
        """
        Do some sanity checking on the incoming Space:
        Must not be Container (for now) and must have a batch rank.
        """
        for api_method_name, in_spaces in input_spaces.items():
            assert api_method_name == "apply" or api_method_name == "_variables", \
                "ERROR: API for NN-Layer must be `apply|_variables`! No other API-methods allowed."
            if api_method_name == "apply":
                for in_space in in_spaces:
                    # - NNLayers always need FloatBoxes as input (later, add Container Spaces containing only
                    # FloatBoxes).
                    # - Must have batch rank.
                    sanity_check_space(in_space, allowed_types=[FloatBox], must_have_batch_rank=True)

    def _graph_fn_apply(self, *inputs):
        """
        The actual calculation on one or more input Ops.

        Args:
            inputs (SingleDataOp): The single (non-container) input(s) to the layer.

        Returns:
            The output(s) after having pushed the input(s) through the layer.
        """
        return self.layer.apply(*inputs)
