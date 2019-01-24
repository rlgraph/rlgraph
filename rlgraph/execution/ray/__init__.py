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

from rlgraph.execution.ray.ray_executor import RayExecutor
from rlgraph.execution.ray.ray_value_worker import RayValueWorker

from rlgraph.execution.ray.apex import ApexExecutor, ApexMemory, RayMemoryActor
from rlgraph.execution.ray.sync_batch_executor import SyncBatchExecutor

RayExecutor.__lookup_classes__ = dict(
    apex=ApexExecutor,
    apexecutor=ApexExecutor,
    syncbatch=SyncBatchExecutor,
    syncbatchexecutor=SyncBatchExecutor
)

__all__ = ["RayExecutor", "RayValueWorker", "ApexExecutor", "ApexMemory", "RayMemoryActor"]
