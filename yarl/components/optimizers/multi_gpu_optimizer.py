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

from yarl import get_backend
from yarl.components import Component

if get_backend() == "tf":
    import tensorflow


class MultiGpuOptimizer(Component):
    """
    The Multi-GPU optimizer parallelizes optimization across multipe gpus.
    """
    def __init__(self, local_optimizer=None, scope="multi-gpu-optimizer", **kwargs):
        """
        Args:
            local_optimizer (Optional[dict,LocalOptimizer]): The spec-dict for the LocalOptimizer object used
            to run on each device or a LocalOptimizer object itself.
        """
        super(MultiGpuOptimizer, self).__init__(scope=scope, **kwargs)
        self.optimizer = local_optimizer

    def create_variables(self, input_spaces, action_space):
        super(MultiGpuOptimizer, self).create_variables(input_spaces, action_space)

        # TODO setup variables per device

        # TODO assert correct scope and device assignment

    def _graph_fn_load_to_device(self, *inputs):
        """
        Loads inputs to device memories by splitting data across configured devices.

        Args:
            *inputs (SingleDataOp): Data batch.

        Returns:
            int: Tuples per device
        """
        # TODO generate ops to load to device memory
        pass

    def _graph_fn_init_ops(self, *inputs):
        """

        Args:
            *inputs:

        Returns:

        """
        # TODO init loss ops
        pass

