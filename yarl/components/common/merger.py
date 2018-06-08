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


class Merger(Component):
    """
    Merges incoming items into one FlattenedDataOp.
    """
    def __init__(self, output_space, scope="merger", **kwargs):
        """
        Args:
            output_space (Space): The output Space to merge to from the single components. Must be a ContainerSpace.
        """
        assert isinstance(output_space, ContainerSpace), "ERROR: `output_space` must be a ContainerSpace " \
                                                         "(Dict or Tuple)!"
        # We are merging already SingleDataOps: Do not flatten.
        super(Merger, self).__init__(scope=scope, flatten_ops=kwargs.pop("flatten_ops", False), **kwargs)

        self.output_space = output_space
        assert isinstance(output_space, ContainerSpace),\
            "ERROR: `output_space` of Merger Component must be a ContainerSpace (but is {})!".format(output_space)

        # Define the interface (one input, many outputs named after the auto-keys generated).
        self.define_outputs("output")
        self.input_names = list(output_space.flatten().keys())
        self.define_inputs(*self.input_names)
        # Insert our merging GraphFunction.
        self.add_graph_fn(self.input_names, "output", self._graph_fn_merge, flatten_ops=False)

    def _graph_fn_merge(self, *inputs):
        """
        Merges the inputs into a single FlattenedDataOp.

        Args:
            *inputs (DataOp): The input items to be merged back into a FlattenedDataOp.

        Returns:
            FlattenedDataOp: The FlattenedDataOp as a merger of all inputs.
        """
        ret = FlattenedDataOp()
        for i, op in enumerate(inputs):
            ret[self.input_names[i]] = op

        return ret

