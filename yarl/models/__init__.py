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

from .model import Model
from .tensorflow_model import TensorFlowModel


def backend_model(backend="tf"):
    """
    Returns default class for backend.
    Args:
        backend (str): Backend string, e.g. "tf" for TensorFlow.

    Returns: Model class for the specified backend.
    """
    if backend == "tf":
        return TensorFlowModel


__lookup_classes__ = dict(
    tfmodel=TensorFlowModel,
    tensorflowmodel=TensorFlowModel,
    tf=TensorFlowModel,
    tensorflow=TensorFlowModel
)

__all__ = ["Model", "TensorFlowModel", "backend_model"]
