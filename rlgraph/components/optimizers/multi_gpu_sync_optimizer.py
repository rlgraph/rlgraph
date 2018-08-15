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
from rlgraph.components.optimizers.optimizer import Optimizer
from rlgraph.spaces import Dict

if get_backend() == "tf":
    import tensorflow as tf


class MultiGpuSyncOptimizer(Optimizer):
    """
    The Multi-GPU optimizer parallelizes synchronous optimization across multiple GPUs.
    """
    def __init__(self, local_optimizer, devices, scope="multi-gpu-sync-optimizer", **kwargs):
        """
        Args:
            local_optimizer (Optimizer): Local optimizer object to wrap with the multi-gpu optimizer.
            devices (list):
        """
        super(MultiGpuSyncOptimizer, self).__init__(scope=scope, **kwargs)

        # Add local Optimizer object.
        self.local_optimizer = local_optimizer
        # TODO necessary?
        self.add_components(local_optimizer)

        self.gpu_devices = devices
        self.num_gpus = len(devices)
        assert self.num_gpus > 1, "ERROR: The MultiGPUSyncOptimizer requires as least two GPUs but only {} " \
                                  "device ids were passed in.".format(self.num_gpus)
        self.add_components(self.local_optimizer)

        # Function handle used to create replicas.
        self.subgraphs = None

        # Device names and variables.
        self.gpu_devices = None
        self.sub_graph_vars = None

        def step(self):
            grads_and_vars = self.call(self._graph_fn_calculate_gradients)
            return self.call(self._graph_fn_apply_gradients, grads_and_vars)

        self.define_api_method("step", step)
        self.define_api_method("load_to_device", self._graph_fn_load_to_device, flatten_ops=True,
                               split_ops=True, add_auto_key_as_first_param=True)

    def set_replicas(self, component_graphs, dict_splitter):
        """
        Provides the optimizer with a list of sub-graphs to use for splitting batches over GPUs.

        Args:
            component_graphs (list): List of component graphs.
            dict_splitter (DictSplitter): Splitter object containing the keys needed to split an input batch into
                the shards for each device.
        """
        self.subgraphs = component_graphs
        self.add_components(*component_graphs)
        self.dict_splitter = dict_splitter

    def create_variables(self, input_spaces, action_space=None):
        super(MultiGpuSyncOptimizer, self).create_variables(input_spaces, action_space)

        # Get input space to load device fun.
        device_input_space = dict()
        for name, space in input_spaces.items():
            # TODO more elegant approach to fetch input space for these tuple spaces?
            if name.startswith("device_inputs"):
                device_input_space[name] = space

        # Turn into container space for easy variable creation.
        self.device_input_space = Dict.from_spec(spec=device_input_space)
        self.sub_graph_vars = list()

        # Create input variables for devices.
        for device in self.gpu_devices:
            with tf.device(device):
                device_variable = self.get_variable(
                    name="gpu_placeholder_{}".format(str(device)), trainable=False,
                    from_space=self.device_input_space,
                    # TODO false or True?
                    flatten=True,
                    add_batch_rank=True,
                    initializer=0
                )
                self.sub_graph_vars.append(device_variable)

    def _graph_fn_load_to_device(self, key, *device_inputs):
        """
        Loads inputs to device memories by splitting data across configured devices.

        Args:

            *DataOpTuple (DataOpTuple): Data batch.

        Returns:
            DataOpTuple: Identity op of device allocated variables.
        """
        if get_backend() == "tf":
            # Assign shard values to device.
            shard_vars = []
            for i, shard in enumerate(device_inputs):
                self.sub_graph_vars[i][key] = device_inputs[i]
                shard_vars.append(self.sub_graph_vars[i][key] )
            return tuple(shard_vars)

    def _graph_fn_calculate_gradients(self, input_batches):
        """
        The multi-gpu-sync optimizer calculates gradients by averaging them across
        replicas.

        Args:
            input_batches(list): List of FlattenedDataOps containing input batch shard per item.
        """
        all_grads_and_vars = list()
        assert len(input_batches) == self.num_gpus
        for i, shard in enumerate(input_batches):
            device_inputs = self.dict_splitter.call("split", shard)

            # Fetch optimizer for this subgraph.
            sub_graph_opt = self.subgraphs[i].sub_component_by_name("optimizer")

            # Obtain gradients for this shard.
            tower_grads = self.call(sub_graph_opt.calculate_gradients, *device_inputs)
            all_grads_and_vars.append(tower_grads)

        # Return averaged gradients.
        return self._average_gradients(all_grads_and_vars)

    def _graph_fn_apply_gradients(self, grads_and_vars):
        """
        These should be the averaged gradients across devices. From the perspective of the
        user of this wrapped optimizer, the API does not change.
        """
        self.local_optimizer._graph_fn_apply_gradients(grads_and_vars=grads_and_vars)

    @staticmethod
    def _average_gradients(gpu_gradients):
        """
        Utility to average gradients across replicas.

        Note: ported from Ray RLLib to demonstrate the different modularization in RLGraph.

        Args:
            gpu_gradients (list): List grads_and_vars lists.

        Returns:
            list: List of grads_and_vars tuples averaged across GPUs.
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
