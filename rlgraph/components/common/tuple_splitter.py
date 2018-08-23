# Copyright 2018 The RLgraph authors. All Rights Reserved.
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

from rlgraph.utils.rlgraph_error import RLGraphError
from rlgraph.components import Component
from rlgraph.spaces import Tuple


class TupleSplitter(Component):
    """
    Splits an incoming container Space into all its single primitive Spaces.
    """
    def __init__(self, tuple_length, **kwargs):
        """
        Args:
            tuple_length (int): The length of the Tuple Space to split.
        """
        raise RLGraphError("TupleSplitter is an obsoleted class! Use ContainerSplitter instead.")
        self.tuple_length = tuple_length

        super(TupleSplitter, self).__init__(
            scope=kwargs.pop("scope", "tuple-splitter"),
            graph_fn_num_outputs=dict(_graph_fn_split=self.tuple_length),
            **kwargs
        )

        self.define_api_method(name="split", func=self._graph_fn_split)

    def check_input_spaces(self, input_spaces, action_space=None):
        in_space = input_spaces["inputs"]

        # Make sure input is a Tuple.
        assert isinstance(in_space, Tuple), "ERROR: Input Space for TupleSplitter ({}) must be Tuple (but is {})!".\
            format(self.global_scope, in_space)

        # Length of the incoming tuple must correspond with given lengths in ctor.
        if len(in_space) != self.tuple_length:
            raise RLGraphError("Length of input Space ({}) != `self.tuple_length` ({})! Input space is '{}'.".
                               format(len(in_space), self.tuple_length, in_space))

    def _graph_fn_split(self, inputs):
        """
        Splits the inputs at 0th level into the Spaces at that level (may still be ContainerSpaces in returned
        values).

        Args:
            inputs (DataOpTuple): The input Tuple to be split at 0th level.

        Returns:
            tuple: The tuple of the sub-Spaces (may still be Containers) sorted the same way as they were in the tuple.
        """
        return tuple([value for value in inputs])
