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

import re

from rlgraph import get_backend
from rlgraph.components.common.batch_splitter import BatchSplitter
from rlgraph.components.component import Component
from rlgraph.spaces import Dict
from rlgraph.utils.decorators import rlgraph_api
from rlgraph.utils.ops import DataOpTuple

if get_backend() == "tf":
    import tensorflow as tf


class MultiGpuSynchronizer(Component):
    """
    The Multi-GPU optimizer parallelizes synchronous optimization across multiple GPUs.
    Serves as a replacement pipeline for an Agent's `update_from_external_batch` method, which
    needs to be rerouted through this Component's `calculate_update_from_external_batch` method.
    """
    def __init__(self, batch_size, scope="multi-gpu-sync-optimizer", **kwargs):
        """
        Args:
            batch_size (int): The batch size that will need to be split between the different GPUs
                (each GPU will receive a shard of this batch).
        """
        super(MultiGpuSynchronizer, self).__init__(graph_fn_num_outputs=dict(
            _graph_fn_calculate_update_from_external_batch=4  # TODO: <- This is currently hardcoded for DQN-type agents
        ), scope=scope, **kwargs)

        self.batch_size = batch_size
        self.shard_size = 0

        # The list of GPU-towers (copies of the original agent root-component) that are sub-Components of this one.
        self.towers = None
        self.batch_splitter = None

        # Device names and variables.
        self.gpu_devices = None
        self.num_gpus = 0

        self.tower_placeholders = list()
        self.device_input_space = None

    def setup_towers(self, towers, devices):
        """
        Provides the optimizer with sub-graphs, batch splitting, name of the loss to split over,
        and devices to split over.

        Args:
            towers (list): List of GPU-towers (copies of the original root-component).
            devices (list): List of device names.
        """
        self.gpu_devices = devices
        self.num_gpus = len(devices)
        assert self.num_gpus > 1,\
            "ERROR: The MultiGPUSyncOptimizer requires as least two GPUs but only {} device ids were passed " \
            "in.".format(self.num_gpus)
        self.shard_size = int(self.batch_size / self.num_gpus)

        # Add our GPU-towers (copies of the original agent root-component).
        self.towers = towers
        self.add_components(*self.towers)

        # Splits input shards of `update_from_external_batch`.
        self.batch_splitter = BatchSplitter(self.num_gpus, self.shard_size)
        self.add_components(self.batch_splitter)

    # TODO: Solve this problem via staging areas (one per GPU).
    # TODO: Stuff coming from the batch-splitter can then be staged/unstaged (previous batch shard).
    def create_variables(self, input_spaces, action_space=None):
        super(MultiGpuSynchronizer, self).create_variables(input_spaces, action_space)

        # Get input space to load device fun.
        device_input_space = {}
        idx = 0
        while True:
            key = "inputs[{}]".format(idx)
            if key not in input_spaces:
                break
            device_input_space[str(idx)] = input_spaces[key]
            idx += 1
        # Turn into container space for easy variable creation.
        self.device_input_space = Dict(device_input_space)

        # Create input variables for devices.
        for i, device in enumerate(self.gpu_devices):
            with tf.device(device):
                device_variable = self.get_variable(
                    name="gpu-placeholder-{}".format(i),
                    trainable=False,
                    from_space=self.device_input_space,
                    flatten=True,
                    add_batch_rank=self.shard_size,
                    initializer=0
                )
                self.tower_placeholders.append(tuple(device_variable.values()))

    def sync_target_qnets(self):
        tower_ops = list()
        for i in range(self.num_gpus):
            op = self.towers[i].sync_target_qnet()
            tower_ops.append(op)
        group_op = self._graph_fn_group(*tower_ops)
        return group_op

    def _graph_fn_group(self, *tower_ops):
        return tf.group(*tower_ops)

    @rlgraph_api
    def _graph_fn_calculate_update_from_external_batch(self, main_variables, *inputs):
        # - Init device memory, i.e. load batch to GPU memory.
        # - Call gradient calculation on multi-gpu optimizer which splits batch
        #   and gets gradients from each sub-graph, then averages them.
        # - Apply averaged gradients to master component.
        # - Sync new weights to towers.

        # Split the incoming batch into its per-GPU shards.
        input_batches = self.batch_splitter.split_batch(*inputs)

        # Load shards to the different devices.
        per_device_assign_ops, loaded_input_batches = self._load_to_device(*input_batches)

        all_grads_and_vars = []
        all_loss = []
        all_loss_per_item = []
        all_rest = None

        assert len(loaded_input_batches) == self.num_gpus
        for i, shard_data in enumerate(loaded_input_batches):
            with tf.control_dependencies([per_device_assign_ops[i]]):
                shard_data_stopped = tuple([tf.stop_gradient(datum.read_value()) for datum in shard_data])
                return_values_to_be_averaged = self.towers[i].update_from_external_batch(*shard_data_stopped)

                grads_and_vars = return_values_to_be_averaged[0]
                loss = return_values_to_be_averaged[1]
                loss_per_item = return_values_to_be_averaged[2]
                rest = return_values_to_be_averaged[3:]
                if all_rest is None:
                    all_rest = [list()] * len(rest)

                all_grads_and_vars.append(grads_and_vars)
                all_loss.append(loss)
                all_loss_per_item.append(loss_per_item)
                for j, r in enumerate(rest):
                    all_rest[j].append(r)

        ret = [
            # Average over the gradients per variable.
            self._average_grads_and_vars(main_variables, all_grads_and_vars),
            # Simple average over all GPUs.
            tf.reduce_mean(tf.stack(all_loss, axis=0)),
            # concatenate the loss_per_item to regenerate original (un-split) batch
            tf.concat(all_loss_per_item, axis=0),
        ]

        # For the remaining return items, do like for loss-per-item (regenerate values for original, unsplit batch).
        for rest_list in all_rest:
            ret.append(tf.concat(rest_list, axis=0))

        # Return averaged gradients.
        return tuple(ret)

    @rlgraph_api(must_be_complete=False)
    def _graph_fn_sync_policy_weights_to_towers(self, optimizer_step_op, policy_variables):
        # Wait for the optimizer update, then sync all variables from the main (root) policy to each tower.
        with tf.control_dependencies([optimizer_step_op]):
            sync_ops = []
            for i, tower in enumerate(self.towers):
                # Sync weights to shards
                sync_op = self.towers[i].set_policy_weights(policy_variables)
                sync_ops.append(sync_op)

            return tf.group(*sync_ops)

    def _load_to_device(self, *device_inputs):
        """
        Loads inputs to device memories by splitting data across configured devices.

        Args:
            *device_inputs (Tuple[DataOpTuple]): One or more DataOpTuples, each one representing the data for a single
                GPU device.

        Returns:
            Tuple[Tuple[DataOpTuple]]:
                - Tuple of assign-ops: One for each GPU.
                - Tuple: The device allocated variables ("GPU placeholder" vars).
        """
        if get_backend() == "tf":
            # Assign shard values to device.
            per_device_assign_ops = []
            for gpu, shard in enumerate(device_inputs):
                assign_ops = []
                for i, var in enumerate(self.tower_placeholders[gpu]):
                    assign_op = tf.assign(var, shard[i])
                    assign_ops.append(assign_op)
                per_device_assign_ops.append(tf.group(*assign_ops, name="load-placeholders-gpu{}".format(gpu)))

            return tuple(per_device_assign_ops), tuple(self.tower_placeholders)

    def _average_grads_and_vars(self, main_variables, grads_and_vars_all_gpus):
        """
        Utility to average gradients (per var) across towers.

        Note: ported from Ray RLLib to demonstrate the different modularization in RLGraph.

        Args:
            grads_and_vars (list): List grads_and_vars lists.

        Returns:
            list: List of grads_and_vars tuples averaged across GPUs.
        """
        gpu_grad_averages = []
        if get_backend() == "tf":
            for i, grads_and_vars in enumerate(zip(*grads_and_vars_all_gpus)):
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
                # Need the actual main policy vars, as these are the ones that should be updated.
                # TODO: This is a hack and needs to be changed, but it works for now to look up main policy variables.
                main_variable_key = re.sub(r'{}/tower-0/'.format(self.global_scope), "", grads_and_vars[0][1].op.name)
                main_variable_key = re.sub(r'/', "-", main_variable_key)
                var = main_variables[main_variable_key]
                gpu_grad_averages.append((mean_grad, var))

        return DataOpTuple(gpu_grad_averages)
