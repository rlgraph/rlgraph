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

from rlgraph.components.component import Component
from rlgraph.spaces.space import Space
from rlgraph.utils.decorators import rlgraph_api


class Layer(Component):
    """
    A Layer is a simple Component that implements the `call` method with n inputs and m return values.
    """
    def __init__(self, scope="layer", **kwargs):
        # Assume 1 return value when calling `call`.
        graph_fn_num_outputs = kwargs.pop("graph_fn_num_outputs", {"_graph_fn_call": 1})
        super(Layer, self).__init__(scope=scope, graph_fn_num_outputs=graph_fn_num_outputs, **kwargs)

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
    def _graph_fn_call(self, *inputs):
        """
        Applies the layer's logic to the inputs and returns one or more result values.

        Args:
            *inputs (any): The input(s) to this layer.

        Returns:
            DataOp: The output(s) of this layer.
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """
        Make all Layers callable for the Keras-style functional API.

        Args:
            *args ():

        Returns:

        """
        # If Spaces are given, add this information already to `self.api_method_records`.
        for i, arg in enumerate(args):
            if isinstance(arg, Space):
                if self.api_methods["call"].args_name is None:
                    self.api_method_inputs[self.api_methods["call"].input_names[i]] = arg
                else:
                    self.api_method_inputs["{}[{}]".format(self.api_methods["call"].args_name, i)] = arg

        for key, value in kwargs.items():
            if isinstance(value, Space):
                self.api_method_inputs[key] = value

        # Need to return as many return values as `call` returns.
        num_outputs = self.graph_fn_num_outputs.get("_graph_fn_call", 1)
        if num_outputs > 1:
            return tuple([
                LayerCallOutput(list(args) + list(kwargs.values()), self, output_slot=i, num_outputs=num_outputs)
                for i in range(num_outputs)
            ])
        else:
            return LayerCallOutput(list(args) + list(kwargs.values()), self)


class LayerCallOutput(object):
    def __init__(self, inputs, component, output_slot=0, num_outputs=1):
        self.inputs = inputs
        self.component = component
        self.output_slot = output_slot
        self.num_outputs = num_outputs
        self.var_name = None

    def __lt__(self, other):
        return self.output_slot < other.output_slot

    def __gt__(self, other):
        return self.output_slot > other.output_slot
