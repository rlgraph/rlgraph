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

from yarl.components import Component, EXPOSE_INS, EXPOSE_OUTS


class StackComponent(Component):
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
            sub_components (List[Component]): The sub-components to immediately place into this one and
                connect to each other.

        Keyword Args:
            expose_ins (bool): Whether to expose the first sub-component's inputs (default: True).
            expose_outs (bool): Whether to expose the last sub-component's outputs (default: True).

        Raises:
            YARLError: If sub-components' number of inputs/outputs do not match.
        """
        expose_ins = kwargs.pop("expose_ins", True)
        expose_outs = kwargs.pop("expose_outs", True)

        super(StackComponent, self).__init__(**kwargs)
        # An empty Stack is the same as a base Component object.
        if len(sub_components) == 0:
            return

        # Add all sub-components into the stack obeying the settings for connecting the first component's input
        # Sockets and the last Component's output Socket.
        self.add_components(*sub_components, expose=dict({
            sub_components[0].name: EXPOSE_INS if expose_ins else False,
            sub_components[-1].name: EXPOSE_OUTS if expose_outs else False
        }))

        # Now connect all sub-components with each other.
        for i in range(len(sub_components)-1):
            component = sub_components[i]
            next_component = sub_components[i+1]

            # TODO: change this to make it possible to just pass in two components, and they would be connected
            # connect component's output socket(s) with next_component's input socket(s).
            for out_sock, in_sock in zip(component.output_sockets, next_component.input_sockets):
                # correctly with each other if the first one has the same number of
                # outputs than the second one has inputs.
                self.connect([component, out_sock], [next_component, in_sock])

