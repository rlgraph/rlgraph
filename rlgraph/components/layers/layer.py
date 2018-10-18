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

from rlgraph.components.component import Component
from rlgraph.utils.decorators import rlgraph_api


class Layer(Component):
    """
    A Layer is a simple Component that implements the `apply` method with n inputs and m return values.
    """
    def __init__(self, **kwargs):
        super(Layer, self).__init__(scope=kwargs.pop("scope", "layer"), **kwargs)

    def get_preprocessed_space(self, space):
        """
        Returns the Space obtained after pushing the space input through this layer.

        Args:
            space (Space): The incoming Space object.

        Returns:
            Space: The Space after preprocessing.
        """
        return space

    @rlgraph_api
    def _graph_fn_apply(self, *inputs):
        """
        Applies the layer's logic to the inputs and returns one or more result values.

        Args:
            *inputs (any): The input(s) to this layer.

        Returns:
            DataOp: The output(s) of this layer.
        """
        raise NotImplementedError

