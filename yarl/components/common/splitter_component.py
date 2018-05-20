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
from yarl.spaces import ContainerSpace


class SplitterComponent(Component):
    """
    Splits an incoming container Space into all its single primitive Spaces.
    """
    def __init__(self, input_space, scope="splitter", output_names=None, **kwargs):
        """
        Args:
            input_space (Space): The input Space to split into its single components. Must be a ContainerSpace.
            output_names (List[str]): An optional list of out-Socket names to be used instead of the auto-generated keys
                coming from the flattening operation.
        """
        assert isinstance(input_space, ContainerSpace), "ERROR: `input_space` must be a ContainerSpace (Dict or Tuple)!"
        super(SplitterComponent, self).__init__(scope=scope, **kwargs)

        self.input_space = input_space

        # Define the interface (one input, many outputs named after the auto-keys generated).
        self.define_inputs("input")
        flat_dict = input_space.flatten()
        if output_names is not None:
            assert len(flat_dict) == len(output_names), "ERROR: Number if given out-names ({}) does not " \
                                                        "match number of elements in input " \
                                                        "ContainerSpace ({})!". \
                format(len(output_names), len(flat_dict))
        else:
            output_names = [key for key in flat_dict.keys()]
        self.define_outputs(*output_names)
        # Insert our simple splitting computation.
        self.add_computation("input", output_names, self._computation_split)

    def _computation_split(self, inputs):
        """
        Splits our inputs into all its primitive Spaces in the "right" order. Returns n single ops.

        Args:
            inputs (OrderedDict): The flattened inputs (each value is one primitive op).

        Returns:
            tuple: The (ordered) tuple of the primitive Spaces.
        """
        ret = list()
        for op in inputs.values():
            ret.append(op)

        return tuple(ret)

