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

from rlgraph import get_backend
from rlgraph.graphs import GraphExecutor
from rlgraph.utils import util

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

    def build(self, root_components, input_spaces, *args):
        start = time.perf_counter()
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

    def execute(self, *api_methods):
        # Have to call each method separately.
        ret = list()
        for api_method in api_methods:
            if api_method is None:
                continue
            elif isinstance(api_method, (list, tuple)):
                params = util.force_list(api_method[1])
                api_method = api_method[0]
                api_ret = self.graph_builder.execute_eager_op(api_method, params)
                ret.append(api_ret)

        # Unwrap if len 1.
        ret = ret[0] if len(ret) == 1 else ret
        return ret

    def read_variable_values(self, variables):
        pass

    def init_execution(self): \
        # Nothing to do here for PyTorch.
        pass

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

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass

    def terminate(self):
        pass
