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
from yarl.components.optimizers.optimizer import Optimizer


if get_backend() == "tf":
    import tensorflow


class MultiGpuSyncOptimizer(Optimizer):
    """
    The Multi-GPU optimizer parallelizes synchronous optimization across multipe gpus.
    """

    def __init__(self, local_optimizer=None, scope="multi-gpu-sync-optimizer", **kwargs):
        """
        Args:
            local_optimizer (Optional[dict,LocalOptimizer]): The spec-dict for the LocalOptimizer object used
            to run on each device or a LocalOptimizer object itself.
        """
        super(MultiGpuSyncOptimizer, self).__init__(scope=scope, **kwargs)
        self.optimizer = local_optimizer

        # Function handle used to create replicas.
        self.replica_graph_handle = None

        # Device names.
        self.gpu_devices = None

        # Graph replicas holding the subgraph copies.
        self.device_graphs = None
        self.device_gradients = None
        self.device_vars = None
        self.device_losses = None

    def set_replica_graph_handle(self, build_loss_graph):
        """
        Provides the optimizer with a function generate replicas of the loss graph.

        Args:
            build_loss_graph (Callable): Function returning the loss graph when called.
        """
        self.replica_graph_handle = build_loss_graph

    def set_devices(self, gpu_devices):
        """
        Provides the optimizer with device identifiers to be used for assigning replicas
        to devices.

        Args:
            gpu_devices (list): List of device names.
        """
        self.gpu_devices = gpu_devices

    def create_variables(self, input_spaces, action_space):
        super(MultiGpuSyncOptimizer, self).create_variables(input_spaces, action_space)

        # Create device copies and variables
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

    def _graph_fn_calculate_gradients(self, variables, loss, *inputs):
        """
        The multi-gpu-sync optimizer calculates gradients by averaging them across
        replicas.
        """
        grads = []
        for device in self.gpu_devices:
            grads.append(self.optimizer._graph_fn_calculate_gradients(
                variables=self.device_vars[device],
                loss=self.device_losses[device]
            ))
        return self._average_gradients(grads)

    def _graph_fn_apply_gradients(self, grads_and_vars):
        """

        """
        pass

    def _average_gradients(self, gpu_gradients):
        """
        Utility to average gradients across replicas.

        Note: ported from Ray RLLib to demonstrate the different modularization in YARL.

        Args:
            gpu_gradients (list): List grads_and_vars lists.

        Returns:
            list: List of grads_and_vars tuples averaged across gpus.
        """
        pass
