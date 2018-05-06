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

from yarl.components.component import Component


class StackComponent(Component):
    def __init__(self, *sub_components, **kwargs):
        """
        A component container stack that incorporates one or more sub-components which are all automatically connected
        to each other (in the sequence they are given in the constructor) and then exposed as this component's
        interface as follows: The input(s) of the very first sub-component and the output(s) of the last sub
        component.

        All sub-components need to match in the number of input and output Sockets. E.g. the third sub-component's
        number of outputs has to be the same as the forth sub-component's number of inputs.

        Args:
            sub_components (List[Component]): The sub-components to immediately place into this one and
                connect to each other.

        Raises:
            YARLError: If sub-components' number of inputs/outputs do not match.
        """
        super(StackComponent, self).__init__(**kwargs)

        # Add all sub-components into the stack.
        if len(sub_components) == 0:
            return

        self.add_components(*sub_components)

        # Connect all sub-components with each other.
        for i in range(len(sub_components)-1):
            component = sub_components[i]
            next_component = sub_components[i+1]
            # connect component's output socket(s) with next_component's input socket(s).
            for output_sock, input_sock in zip(component.outputs.values(), next_component.inputs.values()):
                self.connect([component, output_sock], [next_component, input_sock])

        # Expose first sub-component's input(s) as our own.
        # More than one input.
        if len(sub_components[0].inputs) > 1:
            # Connect all 1st sub-component's input sockets to our own.
            for i, input_name in enumerate(sub_components[0].inputs.keys()):
                self.add_sockets(input_name, type_="in")
                self.connect(input_name, [sub_components[0], input_name])
        else:
            # Call our input "input" no matter what the name of the sub-component's in-socket is.
            self.add_sockets("input")
            self.connect("input", sub_components[0].get_socket(type_="in"))

        # Expose last sub-component's output(s) as our own.
        # More than one output.
        if len(sub_components[-1].outputs) > 1:
            # Connect all last sub-component's output sockets to our own.
            for i, output_name in enumerate(sub_components[-1].outputs.keys()):
                self.add_sockets(output_name, type_="out")
                self.connect(output_name, [sub_components[-1], output_name])
        else:
            # Call our input "input" no matter what the name of the sub-component's in-socket is.
            self.add_sockets("output")
            self.connect(sub_components[-1].get_socket(type_="out"), "output")

