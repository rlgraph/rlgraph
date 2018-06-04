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

from yarl.components import Component
from yarl.utils.util import force_list


class Stack(Component):
    """
    A component container stack that incorporates one or more sub-components which are all automatically connected
    to each other (in the sequence they are given in the c'tor) and then exposed as this component's
    interface as follows: The input(s) of the very first sub-component and the output(s) of the last sub
    component.
    All sub-components need to match in the number of input and output Sockets. E.g. the third sub-component's
    number of outputs has to be the same as the forth sub-component's number of inputs.
    """
    def __init__(self, *sub_components, **kwargs):
        """
        Args:
            sub_components (Component): The sub-components to add to the Stack and connect to each other.

        Keyword Args:
            expose_ins (bool): Whether to expose the first sub-component's inputs (default: True).
            expose_outs (bool): Whether to expose the last sub-component's outputs (default: True).
            sub_component_inputs (Optional[List[str]]): List of in-Socket names of a sub-Component that should be
                connected to the corresponding out-Sockets (`sub_component_outputs`) of the previous sub-Component.
            sub_component_outputs (Optional[List[str]]): List of out-Socket names of a sub-Component that should be
                connected to the corresponding in-Sockets (`sub_component_inputs`) of the next sub-Component.

        Raises:
            YARLError: If sub-components' number of inputs/outputs do not match.
        """
        expose_ins = kwargs.pop("expose_ins", True)
        expose_outs = kwargs.pop("expose_outs", True)
        sub_component_inputs = force_list(kwargs.pop("sub_component_inputs", None))
        sub_component_outputs = force_list(kwargs.pop("sub_component_outputs", None))

        super(Stack, self).__init__(*sub_components, **kwargs)

        # Redefine sub_components for iteration purposes.
        sub_components = list(self.sub_components.values())

        if len(self.sub_components) > 0:
            # Connect the first component's input(s) to our in-Sockets (same name) and the last Component's output(s)
            # to our output Socket(s).
            if expose_ins is True:
                for in_sock in sub_components[0].input_sockets:
                    self.define_inputs(in_sock.name)
                    self.connect(in_sock.name, in_sock)
            if expose_outs is True:
                for out_sock in sub_components[-1].output_sockets:
                    self.define_outputs(out_sock.name)
                    self.connect(out_sock, out_sock.name)

        # Now connect all sub-components with each other.
        # By specific in/out-Socket names.
        if len(sub_component_inputs) > 0:
            assert len(sub_component_inputs) == len(sub_component_outputs),\
                "ERROR: `sub_component_inputs` () and `sub_component_outputs` () must have the same length!".\
                format(sub_component_inputs, sub_component_outputs)
            for i in range(len(sub_components) - 1):
                for out_sock, in_sock in zip(sub_component_outputs, sub_component_inputs):
                    self.connect((sub_components[i], out_sock), (sub_components[i + 1], in_sock))
        # Or just all out- with in-Sockets.
        else:
            for i in range(len(sub_components) - 1):
                self.connect(sub_components[i], sub_components[i+1])
