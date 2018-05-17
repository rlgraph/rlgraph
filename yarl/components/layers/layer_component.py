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

from yarl.components.layers.stack_component import StackComponent


class LayerComponent(StackComponent):
    """
    A StackComponent that has its own computation logic (e.g. a NN layer) but - just like a StackComponent - can
    also be constructed via nested sub-components that are first automatically connected to each other
    (in the sequence they are given in this c'tor) and then connected to this component's computation unit.
    The final interface will hence consist of the first sub-component's input(s)- and this layer's computation's
    output-socket(s).
    """
    def __init__(self, *sub_components, **kwargs):
        """
        Args:
            sub_components (List[Component]): The sub-components to immediately place into this one and
                connect to each other. The last sub-component is connected to this one's computation unit,
                which then provides the output Sockets of this component.

        Keyword Args:
            comp_inputs (int): The number of parameters that our computation function takes.
            comp_outputs (int): The number of output values that our computation function returns (as tuple)
                given that the number of input parameters is num_computation_inputs.
        """
        self.computation_inputs = kwargs.pop("computation_inputs", 1)
        self.computation_outputs = kwargs.pop("computation_outputs", 1)
        self.split_container_spaces = kwargs.pop("split_container_spaces", True)

        super(LayerComponent, self).__init__(*sub_components, expose_outs=False, **kwargs)

        # No sub-components, just create empty in-Sockets.
        if len(sub_components) == 0:
            if self.computation_inputs > 1:
                for in_ in range(self.computation_inputs):
                    self.add_sockets("input{}".format(in_))
            else:
                self.add_sockets("input")

        # Create our output Sockets.
        if self.computation_outputs > 1:
            for out_ in range(self.computation_outputs):
                self.add_sockets("output{}".format(out_))
        else:
            self.add_sockets("output")

        # Create computation and connect it from our input socket(s) (or last sub-component's output(s)) to our output
        # socket(s).
        # NOTE: Layers always do split computations on complex input Spaces.
        self.add_computation(sub_components[-1].output_sockets if len(sub_components) > 0 else self.input_sockets,
                             self.output_sockets, "apply", split_container_spaces=self.split_container_spaces)

    def _computation_apply(self, *inputs):
        """
        This is where the computation of this layer goes.

        Args:
            *inputs (any): The input(s) to this layer. The number of inputs must match self.computation_inputs.

        Returns:
            The output(s) of this layer. The number of elements in the returned tuple must match self.computation_outputs.
        """
        return inputs  # optional

