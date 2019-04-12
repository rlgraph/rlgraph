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

import copy

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
            *args (any): The args passed in to the layer when "called" via the Keras-style functional API.

        Keyword Args:
            **kwargs (any): The kwargs passed in to the layer when "called" via the Keras-style functional API.

        Returns:
            List[LayerCallOutput]: n LayerCallOutput objects, where n=number of return values of `self._graph_fn_call`.
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

        inputs_list = list(args) + list(kwargs.values())
        if len(inputs_list) > 0 and isinstance(inputs_list[0], Space):
            inputs_list = [LayerCallOutput([], [], self, output_slot=i, num_outputs=len(inputs_list), space=s)
                           for i, s in enumerate(inputs_list)]
        kwarg_strings = ["" for _ in range(len(args))]+[k + "=" for k in kwargs.keys()]

        # Need to return as many return values as `call` returns.
        num_outputs = self.graph_fn_num_outputs.get("_graph_fn_call", 1)
        if num_outputs > 1:
            siblings = [
                LayerCallOutput(inputs_list, kwarg_strings, self, output_slot=i, num_outputs=num_outputs)
                for i in range(num_outputs)
            ]
            for s in siblings:
                s.inputs_needed.extend(siblings)
                s.inputs_needed.remove(s)
            return tuple(siblings)
        else:
            return LayerCallOutput(inputs_list, kwarg_strings, self)


class LayerCallOutput(object):
    def __init__(self, inputs, kwarg_strings, component, output_slot=0, num_outputs=1, space=None):
        """
        Args:
            inputs (list[LayerCallOutput,Space]): The inputs to the `call` method.
            kwarg_strings (List[str]): The kwargs corresponding to each input in `inputs` (use "" for no kwarg).
            component (Component): The Component, whose `call` method returns this output.
            output_slot (int): The position in the return tuple of the call.
            num_outputs (int): The over all number of return values that the call returns.
            space(Optional[Space]): The Space this object represents iff at the input arg side of the call.
        """
        self.inputs = inputs
        self.inputs_needed = copy.copy(self.inputs)
        self.kwarg_strings = kwarg_strings
        self.component = component
        self.output_slot = output_slot
        self.num_outputs = num_outputs
        self.space = space

        for i in self.inputs:
            self.inputs_needed.extend(i.inputs_needed)

        self.var_name = None

    #def __eq__(self, other):
    #    if other in self.inputs_needed or self.output_slot != other.output_slot:
    #        return False
    #    return True

    def __lt__(self, other):
        # If `self` is dependent on the `other`, put self first.
        if other in self.inputs_needed:
            return True
        # Otherwise, sort by output-slot.
        return self.output_slot < other.output_slot

    def __gt__(self, other):
        # If `self` is dependent on the `other`, put self first.
        if other in self.inputs_needed:
            return False
        # Otherwise, sort by output-slot.
        return self.output_slot > other.output_slot

    #def __le__(self, other):
    #    if other in self.inputs_needed:
    #        return True
    #    return self.output_slot <= other.output_slot

    #def __ge__(self, other):
    #    if other in self.inputs_needed:
    #        return False
    #    return self.output_slot <= other.output_slot
