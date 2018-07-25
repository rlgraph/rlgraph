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

from rlgraph import RLGraphError
from rlgraph.components import Component
from rlgraph.spaces import Dict


class Splitter(Component):
    """
    Splits an incoming container Space into all its single primitive Spaces.
    """
    def __init__(self, *output_order, **kwargs):
        """
        Args:
            *output_order (str): List of 0th level keys by which the return values of `split` must be sorted.
            Example: output_order=["B", "C", "A"]
            -> split(Dict(B=1, A=2, C=10))
            -> return: list(1, 10, 2), where 1, 10, and 2 are ops
        """
        self.output_order = output_order

        super(Splitter, self).__init__(
            scope=kwargs.pop("scope", "splitter"),
            graph_fn_num_outputs=dict(_graph_fn_split=len(output_order)),
            **kwargs
        )

        self.define_api_method(name="split", func=self._graph_fn_split)

    def check_input_spaces(self, input_spaces, action_space):
        in_space = input_spaces["split"][0]
        # Make sure input is a Dict (unsorted).
        assert isinstance(in_space, Dict), "ERROR: Input Space for Splitter ({}) must be Dict (but is {})!".\
            format(self.global_scope, in_space)
        # Keys of in_space must all be part of `self.output_order`.
        for i, name in enumerate(self.output_order):
            if name not in in_space:
                raise RLGraphError("Item {} in `output_order` of Splitter '{}' is not part of the input Space ({})!".
                                format(i, self.global_scope, in_space))

    def _graph_fn_split(self, input_):
        """
        Splits the input_ at flat-key level `self.level` into the Spaces at that level.

        Args:
            input_ (DataOpDict): The input Dict to be split by its primary keys.

        Returns:
            tuple: The tuple of the primary Spaces (may still be Containers) sorted by `self.output_order`.
        """
        ret = [None] * len(self.output_order)
        for key, value in input_.items():
            ret[self.output_order.index(key)] = value

        return tuple(ret)
