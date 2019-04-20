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

from rlgraph.components.layers.preprocessing.preprocess_layer import PreprocessLayer
from rlgraph.spaces import Dict, Tuple
from rlgraph.utils.decorators import rlgraph_api
from rlgraph.utils.rlgraph_errors import RLGraphError


class ContainerSplitter(PreprocessLayer):
    """
    Splits an incoming ContainerSpace into all its single primitive Spaces.
    """
    def __init__(self, *output_order, **kwargs):
        """
        Args:
            *output_order (Union[str,int]):
                For Dict splitting:
                    List of 0th level keys by which the return values of `split` must be sorted.
                    Example: output_order=["B", "C", "A"]
                    -> split(Dict(A=o1, B=o2, C=o3))
                    -> return: list(o2, o3, o1), where o1-3 are ops
                For Tuple splitting:
                    List of 0th level indices by which the return values of `split` must be sorted.
                    Example: output_order=[0, 2, 1]
                    -> split(Tuple(o1, o2, o3))
                    -> return: list(o1, o3, o2), where o1-3 are ops

        Keyword Args:
            tuple_length (Optional[int]): If no output_order is given, use this number to hint how many
                return values our graph_fn has.
        """
        self.tuple_length = kwargs.pop("tuple_length", None)
        assert self.tuple_length or len(output_order) > 0, \
            "ERROR: one of **kwargs `tuple_length` or `output_order` must be provided in ContainerSplitter " \
            "(for tuples)!"
        num_outputs = self.tuple_length or len(output_order)

        super(ContainerSplitter, self).__init__(
            scope=kwargs.pop("scope", "container-splitter"), graph_fn_num_outputs=dict(_graph_fn_call=num_outputs),
            **kwargs
        )
        self.output_order = output_order

        # Only for DictSplitter, define this convenience API-method:
        if self.output_order is not None and len(self.output_order) > 0 and isinstance(self.output_order[0], str):
            @rlgraph_api(component=self)
            def split_into_dict(self, inputs):
                """
                Same as `call`, but returns a dict with keys.

                Args:
                    inputs ():

                Returns:

                """
                out = self._graph_fn_call(inputs)
                ret = dict()
                for i, key in enumerate(self.output_order):
                    ret[key] = out[i]
                return ret

        # Dict or Tuple?
        self.type = None

    def check_input_spaces(self, input_spaces, action_space=None):
        in_space = input_spaces["inputs"]

        self.type = type(in_space)
        if self.output_order is None or len(self.output_order) == 0:
            # Auto-ordering only valid for incoming Tuples.
            assert self.type == Tuple, \
                "ERROR: Cannot use auto-ordering in ContainerSplitter for input Dict spaces! Only ok for Tuples."
            self.output_order = list(range(len(in_space)))

        # Make sure input is a Dict (unsorted).
        assert self.type == Dict or self.type == Tuple,\
            "ERROR: Input Space for ContainerSplitter ({}) must be Dict or Tuple (but is " \
            "{})!".format(self.global_scope, in_space)

        # Keys of in_space must all be part of `self.output_order`.
        for i, name_or_index in enumerate(self.output_order):
            if self.type == Dict and name_or_index not in in_space:
                raise RLGraphError(
                    "Name #{} in `output_order` ({}) of ContainerSplitter '{}'"
                    " is not part of the input Space "
                    "({})!".format(i, name_or_index, self.scope, in_space)
                )
            elif self.type == Tuple and name_or_index >= len(in_space):
                raise RLGraphError(
                    "Index #{} in `output_order` (value={}) of ContainerSplitter '{}'"
                    " is outside the length of the input "
                    "Space ({})!".format(i, name_or_index, self.scope, in_space)
                )

    def create_variables(self, input_spaces, action_space=None):
        in_space = input_spaces["inputs"]
        self.type = type(in_space)

    @rlgraph_api
    def _graph_fn_call(self, inputs):
        """
        Splits the inputs at 0th level into the Spaces at that level (may still be ContainerSpaces in returned
        values).

        Args:
            inputs (DataOpDict): The input Dict/Tuple to be split by its primary keys or along its indices.

        Returns:
            tuple: The tuple of the sub-Spaces (may still be Containers) sorted by `self.output_order`.
        """
        ret = [None] * len(self.output_order)
        if self.type == Dict:
            for key, value in inputs.items():
                ret[self.output_order.index(key)] = value
        else:
            # No special ordering -> return as is.
            #if self.output_order is None:
            #    for index, value in enumerate(inputs):
            #        ret[index] = value
            ## Custom re-ordering of the input tuple.
            #else:
            for index, value in enumerate(inputs):
                ret[self.output_order.index(index)] = value

        return tuple(ret)
