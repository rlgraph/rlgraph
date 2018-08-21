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

from rlgraph import get_backend
from rlgraph.graphs.meta_graph import MetaGraph
from rlgraph.graphs.meta_graph_builder import MetaGraphBuilder
from rlgraph.graphs.graph_builder import GraphBuilder
from rlgraph.graphs.graph_executor import GraphExecutor
from rlgraph.graphs.pytorch_executor import PyTorchExecutor
from rlgraph.graphs.tensorflow_executor import TensorFlowExecutor


def backend_executor():
    """
    Returns default class for backend.
    Returns: Executioner for the specified backend.
    """
    if get_backend() == "tf":
        return TensorFlowExecutor
    elif get_backend() == "pytorch":
        return PyTorchExecutor


GraphExecutor.__lookup_classes__ = dict(
    tf=TensorFlowExecutor,
    tensorflow=TensorFlowExecutor,
    pt=PyTorchExecutor,
    pytorch=PyTorchExecutor
)

__all__ = ["MetaGraph", "MetaGraphBuilder", "GraphBuilder",
           "GraphExecutor", "TensorFlowExecutor", "PyTorchExecutor", "backend_executor"]
