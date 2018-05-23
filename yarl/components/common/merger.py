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
from yarl.utils.ops import FlattenedDataOp


class MergerComponent(Component):
    """
    Merges incoming items into one FlattenedDataOp.
    """
    def __init__(self, output_space, scope="merger", input_names=None, **kwargs):
        """
        Args:
            output_space (Space): The output Space to merge to from the single components. Must be a ContainerSpace.
            input_names (List[str]): An optional list of in-Socket names to be used instead of the auto-generated keys
                coming from the flattened output.
        """
        assert isinstance(output_space, ContainerSpace), "ERROR: `output_space` must be a ContainerSpace " \
                                                         "(Dict or Tuple)!"
        super(MergerComponent, self).__init__(scope=scope, **kwargs)

        self.output_space = output_space

        # Define the interface (one input, many outputs named after the auto-keys generated).
        self.define_outputs("output")
        flat_dict = output_space.flatten()
        if input_names is not None:
            assert len(flat_dict) == len(input_names), "ERROR: Number of given in-names ({}) does not " \
                                                       "match number of elements in output " \
                                                       "ContainerSpace ({})!". \
                format(len(input_names), len(flat_dict))
        else:
            input_names = [key for key in flat_dict.keys()]
        self.define_inputs(*input_names)
        # Insert our merging computation.
        self.add_computation(input_names, "output", self._computation_merge,
                             flatten_ops=False)

    def _computation_merge(self, *inputs):
        """
        Merges the inputs into a single FlattenedDataOp.

        Args:
            *inputs (DataOp): The input items to be merged back into a FlattenedDataOp.

        Returns:
            FlattenedDataOp: The FlattenedDataOp as a merger of all inputs.
        """
        ret = FlattenedDataOp()
        for i, op in enumerate(inputs):
            ret[self.input_sockets[i].name] = op

        return ret

