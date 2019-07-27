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

import time

import numpy as np

from rlgraph.graphs import GraphExecutor
from rlgraph.utils import util


class PythonExecutor(GraphExecutor):
    """
    Manages execution for component graphs using define-by-run semantics in pure python (numpy).
    """
    def __init__(self, **kwargs):
        super(PythonExecutor, self).__init__(**kwargs)

        self.global_training_timestep = 0

        ##OBSOLETED: Squeeze result dims, often necessary in tests.
        # Should not behave differently from tf/pytorch backends (graph_fns are identical in functionality).
        self.remove_batch_dims = False

    def build(self, root_components, input_spaces, **kwargs):
        start = time.perf_counter()
        self.init_execution()

        meta_build_times = []
        build_times = []
        for component in root_components:
            start = time.perf_counter()

            # Collect all components.
            components = component.get_all_sub_components(exclude_self=False)
            # Point to this GraphBuilder object.
            for c in components:
                c.graph_builder = self.graph_builder

            ## Make sure we check input-spaces and create variables.
            ##component.when_input_complete(input_spaces=input_spaces)
            # Build the meta-graph.
            meta_graph = self.meta_graph_builder.build(component, input_spaces)
            meta_build_times.append(time.perf_counter() - start)

            build_time = self.graph_builder.build_define_by_run_graph(
                meta_graph=meta_graph, input_spaces=input_spaces, available_devices=None
            )
            build_times.append(build_time)

        return dict(
            total_build_time=time.perf_counter() - start,
            meta_graph_build_times=meta_build_times,
            build_times=build_times,
        )

    def execute(self, *api_method_calls):
        # Have to call each method separately.
        ret = []
        for api_method in api_method_calls:
            if api_method is None:
                continue
            elif isinstance(api_method, (list, tuple)):
                # Which ops are supposed to be returned?
                op_or_indices_to_return = api_method[2] if len(api_method) > 2 else None
                params = util.force_list(api_method[1])
                api_method = api_method[0]

                api_ret = self.graph_builder.execute_define_by_run_op(api_method, params)
                is_dict_result = isinstance(api_ret, dict)
                if not isinstance(api_ret, list) and not isinstance(api_ret, tuple):
                    api_ret = [api_ret]
                to_return = []
                if op_or_indices_to_return is not None:
                    # Op indices can be integers into a result list or strings into a result dict.
                    if is_dict_result:
                        if isinstance(op_or_indices_to_return, str):
                            op_or_indices_to_return = [op_or_indices_to_return]
                        result_dict = {}
                        for key in op_or_indices_to_return:
                                result_dict[key] = api_ret[0][key]
                        to_return.append(result_dict)
                    else:
                        # Build return ops in correct order.
                        # TODO clarify op indices order vs tensorflow.
                        for i in sorted(op_or_indices_to_return):
                            op_result = api_ret[i]
                            to_return.append(op_result)

                else:
                    # Just return everything in the order it was returned by the API method.
                    if api_ret is not None:
                        for op_result in api_ret:
                            to_return.append(op_result)

                # Clean and return.
                self.clean_results(ret, to_return)
            else:
                # Api method is string without args:
                to_return = []
                api_ret = self.graph_builder.execute_define_by_run_op(api_method)
                if api_ret is None:
                    continue
                if not isinstance(api_ret, list) and not isinstance(api_ret, tuple):
                    api_ret = [api_ret]
                for op_result in api_ret:
                    to_return.append(op_result)

                # Clean and return.
                self.clean_results(ret, to_return)

        # Unwrap if len 1.
        ret = ret[0] if len(ret) == 1 else ret
        return ret

    def clean_results(self, ret, to_return):
        for result in to_return:
            if isinstance(result, dict):
                cleaned_dict = {k: v for k, v in result.items() if v is not None}
                ret.append(cleaned_dict)
            elif self.remove_batch_dims and isinstance(result, np.ndarray):
                ret.append(np.array(np.squeeze(result)))
            elif hasattr(result, "numpy"):
                ret.append(np.array(result.numpy()))
            else:
                ret.append(result)

    def read_variable_values(self, component, variables):
        # For test compatibility.
        if isinstance(variables, dict):
            ret = {}
            for name, var in variables.items():
                ret[name] = component.read_variable(var)
            return ret
        # Read a list of vars.
        elif isinstance(variables, list):
            return [component.read_variable(var) for var in variables]
        # Attempt to read as single var.
        else:
            return component.read_variable(variables)

    def finish_graph_setup(self):
        # Nothing to do here for PyTorch.
        pass

    def load_model(self, checkpoint_directory=None, checkpoint_path=None):
        pass

    def store_model(self, path=None, add_timestep=True):
        pass

    def get_device_assignments(self, device_names=None):
        pass

    def terminate(self):
        pass
