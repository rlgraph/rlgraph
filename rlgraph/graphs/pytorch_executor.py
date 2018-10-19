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

import os
import time
import numpy as np

from rlgraph import get_backend
from rlgraph.components import Component
from rlgraph.graphs import GraphExecutor
from rlgraph.utils import util
from rlgraph.utils.util import force_torch_tensors

if get_backend() == "pytorch":
    import torch


class PyTorchExecutor(GraphExecutor):
    """
    Manages execution for component graphs using define-by-run semantics.
    """
    def __init__(self, **kwargs):
        super(PyTorchExecutor, self).__init__(**kwargs)

        self.cuda_enabled = torch.cuda.is_available()

        # In PyTorch, tensors are default created on the CPU unless assigned to a visible CUDA device,
        # e.g. via x = tensor([0, 0], device="cuda:0") for the first GPU.
        self.available_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        # TODO handle cuda tensors

        self.default_torch_tensor_type = self.execution_spec.get("dtype", "torch.FloatTensor")
        if self.default_torch_tensor_type is not None:
            torch.set_default_tensor_type(self.default_torch_tensor_type)

        self.torch_num_threads = self.execution_spec.get("torch_num_threads", 1)
        self.omp_num_threads = self.execution_spec.get("OMP_NUM_THREADS", 1)

        # Squeeze result dims, often necessary in tests.
        self.remove_batch_dims = True

    def build(self, root_components, input_spaces, **kwargs):
        start = time.perf_counter()
        self.init_execution()

        meta_build_times = []
        build_times = []
        for component in root_components:
            start = time.perf_counter()
            meta_graph = self.meta_graph_builder.build(component, input_spaces)
            meta_build_times.append(time.perf_counter() - start)

            build_time = self.graph_builder.build_define_by_run_graph(
                meta_graph=meta_graph, input_spaces=input_spaces, available_devices=self.available_devices
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
                op_indices_to_return = api_method[2] if len(api_method) > 2 else None
                params = util.force_list(api_method[1])
                api_method = api_method[0]

                # TODO where to determine this? exec spec?
                requires_grad = False
                if "update" in api_method:
                    requires_grad = True
                tensor_params = force_torch_tensors(params=params, requires_grad=requires_grad)
                api_ret = self.graph_builder.execute_define_by_run_op(api_method, tensor_params)
                if not isinstance(api_ret, list) and not isinstance(api_ret, tuple):
                    api_ret = [api_ret]
                to_return = []
                if op_indices_to_return is not None:
                    # Build return ops in correct order.
                    for i in op_indices_to_return:
                        op_result = api_ret[i]
                        if isinstance(op_result, torch.Tensor) and op_result.requires_grad is True:
                            op_result = op_result.detach()
                        to_return.append(op_result)

                else:
                    # Just return everything in the order it was returned by the API method.
                    if api_ret is not None:
                        for op_result in api_ret:
                            if isinstance(op_result, torch.Tensor) and op_result.requires_grad is True:
                                op_result = op_result.detach()
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
                    if isinstance(op_result, torch.Tensor) and op_result.requires_grad is True:
                        op_result = op_result.detach()
                    to_return.append(op_result)

                # Clean and return.
                self.clean_results(ret, to_return)

        # Unwrap if len 1.
        ret = ret[0] if len(ret) == 1 else ret
        return ret

    def clean_results(self, ret, to_return):
        for result in to_return:
            if self.remove_batch_dims and isinstance(result, np.ndarray):
                ret.append(np.array(np.squeeze(result)))
            elif hasattr(result, "numpy"):
                ret.append(np.array(result.numpy()))
            else:
                ret.append(result)

    def read_variable_values(self, variables):
        # For test compatibility.
        if isinstance(variables, dict):
            ret = {}
            for name, var in variables.items():
                ret[name] = Component.read_variable(var)
            return ret
        elif isinstance(variables, list):
            return [Component.read_variable(var) for var in variables]
        else:
            # Attempt to read as single var.
            return Component.read_variable(variables)

    def init_execution(self): \
        # TODO Import guards here are annoying but otherwise breaks if torch is not installed.
        if get_backend() == "torch":
            torch.set_num_threads(self.torch_num_threads)
            os.environ["OMP_NUM_THREADS"] = str(self.omp_num_threads)

    def finish_graph_setup(self):
        # Nothing to do here for PyTorch.
        pass

    def get_available_devices(self):
        return self.available_devices

    def load_model(self, path=None):
        pass

    def store_model(self, path=None, add_timestep=True):
        pass

    def get_device_assignments(self, device_names=None):
        pass

    def terminate(self):
        pass
