# Copyright 2018/2019 The RLgraph authors. All Rights Reserved.
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

from rlgraph import get_backend
from rlgraph.components.component import Component
from rlgraph.utils.decorators import rlgraph_api
from rlgraph.utils.ops import DataOpDict, DataOpTuple, FLATTEN_SCOPE_PREFIX


class ContainerMerger(Component):
    """
    Merges incoming items into one FlattenedDataOp.
    """
    def __init__(self, *input_names_or_num_items, **kwargs):
        """
        Args:
            *input_names_or_num_items (Union[str,int]): List of the names of the different inputs in the
                order they will be passed into the `merge` API-method in the returned merged Dict.
                Or the number of items in the Tuple to be merged.
                Example:
                input_names_or_num_items = ["A", "B"]
                - merge(Dict(c=1, d=2), Tuple(3, 4))
                - returned value: Dict(A=Dict(c=1, d=2), B=Tuple(3, 4))
                input_names_or_num_items = 3: 3 items will be merged into a Tuple.

        Keyword Args:
            merge_tuples_into_one (bool): Whether to merge incoming DataOpTuples into one single DataOpTuple.
                If True:  tupleA + tupleB -> (tupleA[0] + tupleA[1] + tupleA[...] + tupleB[0] + tupleB[1] ...).
                If False: tupleA + tupleB -> (tupleA + tupleB).

            is_tuple (bool): Whether we should merge a tuple.
        """
        self.merge_tuples_into_one = kwargs.pop("merge_tuples_into_one", False)
        self.is_tuple = kwargs.pop("is_tuple", self.merge_tuples_into_one)

        super(ContainerMerger, self).__init__(scope=kwargs.pop("scope", "container-merger"), **kwargs)

        self.dict_keys = None

        if len(input_names_or_num_items) == 1 and isinstance(input_names_or_num_items[0], int):
            self.is_tuple = True
        else:
            assert all(isinstance(i, str) and not re.search(r'/', i) for i in input_names_or_num_items), \
                "ERROR: Not all input names of DictMerger Component '{}' are strings or some of them have '/' " \
                "characters in them, which are not allowed.".format(self.global_scope)
            self.dict_keys = input_names_or_num_items

    def check_input_spaces(self, input_spaces, action_space=None):
        spaces = []
        idx = 0
        while True:
            key = "inputs[{}]".format(idx)
            if key not in input_spaces:
                break
            spaces.append(input_spaces[key])
            idx += 1

        # If Tuple -> Incoming inputs could be of any number.
        if self.dict_keys:
            len_ = len(self.dict_keys)
            assert len(spaces) == len_,\
                "ERROR: Number of incoming Spaces ({}) does not match number of given `dict_keys` ({}) in" \
                "ContainerMerger Component '{}'!".format(len(spaces), len_, self.global_scope)

    @rlgraph_api
    def _graph_fn_merge(self, *inputs):
        """
        Merges the inputs into a single DataOpDict OR DataOpTuple with the flat keys given in `self.dict_keys`.

        Args:
            *inputs (FlattenedDataOp): The input items to be merged into a ContainerDataOp.

        Returns:
            ContainerDataOp: The DataOpDict or DataOpTuple as a merger of all *inputs.
        """
        if self.is_tuple is True:
            ret = []
            for op in inputs:
                # Merge single items inside a DataOpTuple into resulting tuple.
                if self.merge_tuples_into_one and isinstance(op, DataOpTuple):
                    ret.extend(list(op))
                # Strict by-input merging.
                else:
                    ret.append(op)
            return DataOpTuple(ret)
        else:
            ret = DataOpDict()
            for i, op in enumerate(inputs):
                if get_backend() == "pytorch" and self.execution_mode == "define_by_run":
                    ret[FLATTEN_SCOPE_PREFIX + self.dict_keys[i]] = op
                else:
                    ret[self.dict_keys[i]] = op
            return ret

