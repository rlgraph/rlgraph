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

import re

from rlgraph.components import Component
from rlgraph.utils.decorators import rlgraph_api
from rlgraph.utils.ops import DataOpDict


class DictMerger(Component):
    """
    Merges incoming items into one FlattenedDataOp.
    """
    def __init__(self, *input_names, **kwargs):
        """
        Args:
            *input_names (str): List of the names of the different inputs in the order they will
                be passed into the `merge` API-method in the returned merged Dict.
                Example:
                input_names = ["A", "B"]
                - merge(Dict(c=1, d=2), Tuple(3, 4))
                - returned value: Dict(A=Dict(c=1, d=2), B=Tuple(3, 4))
        """
        super(DictMerger, self).__init__(scope=kwargs.pop("scope", "dict-merger"), **kwargs)

        assert all(isinstance(i, str) and not re.search(r'/', i) for i in input_names), \
            "ERROR: Not all input names of DictMerger Component '{}' are strings or some of them have '/' " \
            "characters in them, which are not allowed.".format(self.global_scope)
        self.input_names = input_names

    def check_input_spaces(self, input_spaces, action_space=None):
        spaces = []
        idx = 0
        while True:
            key = "inputs[{}]".format(idx)
            if key not in input_spaces:
                break
            spaces.append(input_spaces[key])
            idx += 1

        #assert len(spaces) == len(self.input_names),\
        #    "ERROR: Number of incoming Spaces ({}) does not match number of given `input_names` in DictMerger Component " \
        #    "'{}'!".format(len(spaces), len(self.input_names), self.global_scope)

    #    #for space in spaces:
    #    #    assert not isinstance(space, ContainerSpace),\
    #    #        "ERROR: Single Space ({}) going into merger '{}' must not be a Container " \
    #    #        "Space!".format(space, self.global_scope)

    @rlgraph_api
    def _graph_fn_merge(self, *inputs):
        """
        Merges the inputs into a single FlattenedDataOp with the flat keys given in `self.input_names`.

        Args:
            *inputs (FlattenedDataOp): The input items to be merged into a FlattenedDataOp.

        Returns:
            FlattenedDataOp: The FlattenedDataOp as a merger of all api_methods.
        """
        ret = DataOpDict()
        for i, op in enumerate(inputs):
            ret[self.input_names[i]] = op
        return ret

