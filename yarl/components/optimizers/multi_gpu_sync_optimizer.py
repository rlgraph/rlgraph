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
    import tensorflow as tf


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
        for device in self.gpu_devices:
            with tf.device(device):
                with tf.name_scope(self.scope):
                    # TODO split inputs
                    # TODO Create variables
                    # TODO Get gradients for subgraphs
                    pass

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

    def get_device_ops(self, *params):
        # TODO provide init ops.
        fetch_list = []
        feed_dict = dict()
        for i, param in enumerate(params):
            # TODO -> where do we register this api?
            pass
            # placeholder = self.api[api_method][0][i].op
            # if isinstance(placeholder, DataOpTuple):
            #     for ph, p in zip(placeholder, param):
            #         feed_dict[ph] = p
            # else:
            #     feed_dict[placeholder] = param

    def _graph_fn_calculate_gradients(self, variables, loss, *inputs):
        """
        The multi-gpu-sync optimizer calculates gradients by averaging them across
        replicas.
        """
        all_grads_and_vars = []
        for device in self.gpu_devices:
            all_grads_and_vars.append(self.optimizer._graph_fn_calculate_gradients(
                variables=self.device_vars[device],
                loss=self.device_losses[device]
            ))
        return self._average_gradients(all_grads_and_vars)

    def _graph_fn_apply_gradients(self, grads_and_vars):
        """

        """
        pass

    @staticmethod
    def _average_gradients(gpu_gradients):
        """
        Utility to average gradients across replicas.

        Note: ported from Ray RLLib to demonstrate the different modularization in YARL.

        Args:
            gpu_gradients (list): List grads_and_vars lists.

        Returns:
            list: List of grads_and_vars tuples averaged across gpus.
        """
        gpu_averages = []
        if get_backend() == "tf":
            for grads_and_vars in zip(*gpu_gradients):
                gpu_grads = []

                for grad, var in grads_and_vars:
                    if grad is not None:
                        # Add batch dimension.
                        batch_grad = tf.expand_dims(input=grad, axis=0)

                        # Add along axis for that gpu.
                        gpu_grads.append(batch_grad)

                if not gpu_grads:
                    continue

                aggregate_grads = tf.concat(axis=0, values=gpu_grads)
                mean_grad = tf.reduce_mean(input_tensor=aggregate_grads, axis=0)
                # Don't need all vars because they are shared.
                var = grads_and_vars[0][1]
                gpu_averages.append((mean_grad, var))
        return gpu_averages
