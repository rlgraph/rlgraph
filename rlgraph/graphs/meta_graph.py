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

from __future__ import absolute_import, division, print_function

import re

from rlgraph.utils.rlgraph_errors import RLGraphError


class MetaGraph(object):
    """
    Represents a single RLgraph meta graph object.
    """
    def __init__(self, root_component, api, num_ops, build_status=False):
        self.root_component = root_component
        self.api = api
        self.num_ops = num_ops
        self._built = build_status

    def set_to_built(self):
        assert not self._built, "ERROR: Cannot set graph to built if already built."
        self._built = True

    @property
    def build_status(self):
        return self._built

    @staticmethod
    def get_op_rec_from_space(component, space):
        """
        Retrieves an DataOpRecord given a Component and a Space.
        Checks all input args of the Component for the given Space, then retrieves the first best op-rec
        holding that exact Space.

        Args:
            component (Component): The Component whose input-args to check for the given `space`.
            space (Space): The Space to use to find an op-rec.

        Returns:
            DataOpRecord: A DataOpRecord that uses the given `space`.
        """
        for api_input_name, s in component.api_method_inputs.items():
            # Space found -> Use api_input_name to look for a matching op-rec.
            if s is space:
                assert s.id == space.id  # Just make sure.
                for api_method_name, api_method_rec in component.api_methods.items():
                    for i, arg_name in enumerate(api_method_rec.input_names):
                        # Argument found in this API-method -> Return first best matching op-rec.
                        if arg_name == api_input_name or \
                                (api_method_rec.args_name == arg_name and re.match(r'{}\[\d+\]'.format(arg_name), api_input_name)) or \
                                (api_method_rec.kwargs_name == arg_name and re.match(r'{}\[[\'"]\w+[\'"]\]'.format(arg_name), api_input_name)):
                            op_rec = api_method_rec.in_op_columns[0].op_records[i]
                            return op_rec

        raise RLGraphError("No op-rec found in '{}' for Space={}!".format(component.global_scope, space))
