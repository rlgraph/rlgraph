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

import logging
import time

from rlgraph.graphs import MetaGraph
from rlgraph.spaces import Space
from rlgraph.utils import force_list
from rlgraph.utils.op_records import DataOpRecord
from rlgraph.utils.rlgraph_errors import RLGraphError
from rlgraph.utils.specifiable import Specifiable


class MetaGraphBuilder(Specifiable):
    """
    A meta graph builder takes a connected component graph and generates its
    API by building the meta graph.
    """
    def __init__(self):
        super(MetaGraphBuilder, self).__init__()
        self.logger = logging.getLogger(__name__)

    def build(self, root_component, input_spaces=None):
        """
        Builds the meta-graph by constructing op-record columns going into and coming out of all API-methods
        and graph_fns.

        Args:
            root_component (Component): Root component of the meta graph to build.
            input_spaces (Optional[Space]): Input spaces for api methods.
        """

        # Time the meta-graph build:
        DataOpRecord.reset()
        time_start = time.perf_counter()
        api = {}

        # Sanity check input_spaces dict.
        if input_spaces is not None:
            for input_param_name in input_spaces.keys():
                if input_param_name not in root_component.api_method_inputs:
                    raise RLGraphError(
                        "ERROR: `input_spaces` contains an input-parameter name ('{}') that's not defined in any of "
                        "the root-component's ('{}') API-methods, whose args are '{}'!".format(
                            input_param_name, root_component.name, root_component.api_method_inputs
                        )
                    )

        # Call all API methods of the core once and thereby, create empty in-op columns that serve as placeholders
        # and bi-directional links between ops (for the build time).
        for api_method_name, api_method_rec in root_component.api_methods.items():
            self.logger.debug("Building meta-graph of API-method '{}'.".format(api_method_name))

            # Create the loose list of in-op-records depending on signature and input-spaces given.
            # If an arg has a default value, its input-space does not have to be provided.
            in_ops_records = []
            use_named = False
            for i, param_name in enumerate(api_method_rec.input_names):
                # Arg has a default of None (flex). If in input_spaces, arg will be provided.
                if root_component.api_method_inputs[param_name] == "flex":
                    if param_name in input_spaces:
                        in_ops_records.append(
                            DataOpRecord(position=i, kwarg=param_name if use_named else None)
                        )
                    else:
                        use_named = True
                # Already defined (per default arg value (e.g. bool)).
                elif isinstance(root_component.api_method_inputs[param_name], Space):
                    if input_spaces is not None and param_name in input_spaces:
                        in_ops_records.append(DataOpRecord(position=i, kwarg=param_name if use_named else None))
                    else:
                        use_named = True
                # No default values -> Must be provided in `input_spaces`.
                else:
                    # A var-positional param.
                    if root_component.api_method_inputs[param_name] == "*flex":
                        assert use_named is False
                        if param_name in input_spaces:
                            in_ops_records.extend([
                                DataOpRecord(position=i + j) for j in range(len(force_list(input_spaces[param_name])))
                            ])
                    # A keyword param.
                    elif root_component.api_method_inputs[param_name] == "**flex":
                        if param_name in input_spaces:
                            assert use_named is False
                            in_ops_records.extend([
                                DataOpRecord(kwarg=key) for key in sorted(input_spaces[param_name].keys())
                            ])
                        use_named = True
                    else:
                        # TODO: If space not provided in input_spaces -> Try to call this API method later (maybe another API-method).
                        assert input_spaces is not None and param_name in input_spaces, \
                            "ERROR: arg-name '{}' not defined in input_spaces!".format(param_name)
                        in_ops_records.append(DataOpRecord(position=i, kwarg=param_name if use_named else None))

            # Do the actual core API-method call (thereby assembling the meta-graph).
            args = [op_rec for op_rec in in_ops_records if op_rec.kwarg is None]
            kwargs = {op_rec.kwarg: op_rec for op_rec in in_ops_records if op_rec.kwarg is not None}
            getattr(api_method_rec.component, api_method_name)(*args, **kwargs)

            # Register core's interface.
            api[api_method_name] = (in_ops_records, api_method_rec.out_op_columns[-1].op_records)

            # Tag very last out-op-records with is_terminal_op=True, so we know in the build process that we are done.
            for op_rec in api_method_rec.out_op_columns[-1].op_records:
                op_rec.is_terminal_op = True

        time_build = time.perf_counter() - time_start
        self.logger.info("Meta-graph build completed in {} s.".format(time_build))

        # Get some stats on the graph and report.
        num_meta_ops = DataOpRecord._ID + 1
        self.logger.info("Meta-graph op-records generated: {}".format(num_meta_ops))

        return MetaGraph(root_component=root_component, api=api, num_ops=num_meta_ops, build_status=True)
