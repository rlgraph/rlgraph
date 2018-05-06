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

from .stack_component import StackComponent


class LayerComponent(StackComponent):
    def __init__(self, *sub_components, **kwargs):
        """
        A StackComponent that has its own computation logic (e.g. a NN layer) but - just like a StackComponent - can
        also be constructed via nested sub-components that are first automatically connected to each other
        (in the sequence they are given in this constructor) and then connected to this component's computation unit.
        The final interface will hence consist of the first sub-component's input(s)- and this layer's computation's
        output-socket(s).

        Args:
            sub_components (List[Component]): The sub-components to immediately place into this one and
                connect to each other. The last sub-component is connected to this one's computation unit,
                which then provides the output Sockets of this component.

        Keyword Args:
            comp_inputs (int): The number of parameters that our computation function takes.
            comp_outputs (int): The number of output values that our computation function returns (as tuple)
                given that the number of input paramaeters is num_computation_inputs.
        """
        super(LayerComponent, self).__init__(*sub_components, **kwargs)

        self.comp_inputs = kwargs.get("comp_inputs", 1)
        self.comp_outputs = kwargs.get("comp_outputs", 1)

        # We have sub-components.
        if len(sub_components) > 0:
            # Cut all connection(s) again between last sub-comp and our output socket.
            for sub_comp_sock, self_sock in zip(sub_components[-1].outputs.values(), self.output_sockets.values()):
                self.disconnect([sub_components[-1], sub_comp_sock], [self, self_sock])

        # No sub-components, just create empty Sockets.
        else:
            if self.comp_inputs > 1:
                for in_ in range(self.comp_inputs):
                    self.add_sockets("input{}".format(in_))
            else:
                self.add_sockets("input")

        # Create computation and connect it from our input socket(s) (or last sub-component's output(S))
        # to our output socket(s).
        if len(sub_components) > 0:
            self.add_computation(sub_components[-1].outputs.values(), self.output_sockets.values())
        else:
            self.add_computation(self.input_sockets.values(), self.output_sockets.values())

    def _computation_apply(self, *inputs):
        """
        This is where the computation of this layer goes.

        Args:
            *inputs (any): The input(s) to this layer. The number of inputs must match self.comp_inputs.

        Returns:
            The output(s) of this layer. The number of elements in the returned tuple must match self.comp_outputs.
        """
        raise NotImplementedError
