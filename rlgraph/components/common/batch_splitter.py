
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from rlgraph import get_backend
from rlgraph.components import Component
from rlgraph.utils.decorators import rlgraph_api
from rlgraph.utils.ops import FlattenedDataOp, unflatten_op, DataOpTuple

if get_backend() == "tf":
    import tensorflow as tf


class BatchSplitter(Component):
    """
    Splits a number of incoming DataOps along their batch dimension.
    """
    def __init__(self, num_shards, shard_size, **kwargs):
        """
        Args:
            num_shards (int): Number of shards to split the batch dimension into.
            shard_size (int): The number of samples in a per-GPU shard.
        """
        super(BatchSplitter, self).__init__(
            scope=kwargs.pop("scope", "batch-splitter"),
            graph_fn_num_outputs=dict(_graph_fn_split_batch=num_shards),
            **kwargs
        )

        assert num_shards > 1, "ERROR: num shards must be greater than 1 but is {}.".format(
            num_shards
        )
        self.num_shards = num_shards
        self.shard_size = shard_size

    @rlgraph_api(flatten_ops=True)
    def _graph_fn_split_batch(self, *inputs):
        """
        Splits all DataOps in *inputs along their batch dimension into n equally sized shards. The number of shards
        is determined by `self.num_shards` (int) and the size of each shard depends on the incoming batch size with
        possibly a few superfluous items in the batch being discarded
        (effective batch size = num_shards x shard_size).

        Args:
            *input (FlattenedDataOp): Input tensors which must all have the same batch dimension.

        Returns:
            tuple:
                # Each shard consisting of: A DataOpTuple with len = number of input args.
                # - Each item in the DataOpTuple is a FlattenedDataOp with (flat) key (describing the input-piece
                # (e.g. "/states1")) and values being the (now sharded) batch data for that input piece.

                # e.g. return (for 2 shards):
                # tuple(DataOpTuple(input1_flatdict, input2_flatdict, input3_flatdict, input4_flatdict), DataOpTuple([same]))


                List of FlattenedDataOps () containing DataOpTuples containing the input shards.
        """
        if get_backend() == "tf":
            #batch_size = tf.shape(next(iter(inputs[0].values())))[0]
            #shard_size = tf.cast(batch_size / self.num_shards, dtype=tf.int32)

            # Must be evenly divisible so we slice out an evenly divisible tensor.
            # E.g. 203 items in batch with 4 shards -> Only 4 x 50 = 200 are usable.
            usable_size = self.shard_size * self.num_shards

            # List (one item for each input arg). Each item in the list looks like:
            # A FlattenedDataOp with (flat) keys (describing the input-piece (e.g. "/states1")) and values being
            # lists of len n for the n shards' data.
            inputs_flattened_and_split = list()

            for input_arg_data in inputs:
                shard_dict = FlattenedDataOp()
                for flat_key, data in input_arg_data.items():
                    usable_input_tensor = data[:usable_size]
                    shard_dict[flat_key] = tf.split(value=usable_input_tensor, num_or_size_splits=self.num_shards)
                inputs_flattened_and_split.append(shard_dict)

            # Flip the list to generate a new list where each item represents one shard.
            shard_list = list()
            for shard_idx in range(self.num_shards):
                # To be converted into FlattenedDataOps over the input-arg-pieces once complete.
                input_arg_list = list()
                for input_elem in range(len(inputs)):
                    sharded_data_dict = FlattenedDataOp()
                    for flat_key, shards in inputs_flattened_and_split[input_elem].items():
                        sharded_data_dict[flat_key] = shards[shard_idx]
                    input_arg_list.append(unflatten_op(sharded_data_dict))
                # Must store everything as FlattenedDataOp otherwise the re-nesting will not work.
                shard_list.append(DataOpTuple(input_arg_list))

            # Return n values (n = number of batch shards).
            return tuple(shard_list)

