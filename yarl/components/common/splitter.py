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


class Splitter(Component):
    """
    Splits an incoming container Space into all its single primitive Spaces.
    """
    def __init__(self, input_space, scope="splitter", output_names=None, **kwargs):
        """
        Args:
            input_space (Space): The input Space to split into its single components. Must be a ContainerSpace.
            output_names (List[str]): An optional list of out-Socket names to be used instead of
                the auto-generated keys coming from the flattening operation.
        """
        assert isinstance(input_space, ContainerSpace), \
            "ERROR: `input_space` must be a ContainerSpace (Dict or Tuple)!"
        num_outputs = len(input_space.flatten())
        super(Splitter, self).__init__(scope=scope, graph_fn_num_outputs=dict(_graph_fn_split=num_outputs), **kwargs)

        self.define_api_method(name="split", func=self._graph_fn_split, flatten_ops=True)

    def _graph_fn_split(self, input_):
        """
        Splits our api_methods into all its primitive Spaces in the "right" order. Returns n single ops.

        Args:
            input_ (OrderedDict): The flattened api_methods (each value is one primitive op).

        Returns:
            tuple: The (ordered) tuple of the primitive Spaces.
        """
        ret = list()
        for op in input_.values():
            ret.append(op)

        return tuple(ret)

