
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from rlgraph import get_backend
from rlgraph.components import Component
from rlgraph.utils import DataOpTuple

if get_backend() == "tf":
    import tensorflow as tf


class BatchSplitter(Component):
    """
    Splits a number of incoming tensors along their batch dimension.
    """
    def __init__(self, num_shards, **kwargs):
        """
        Args:
            num_shards (int): Number of shards to split the batch dimension into.
        """
        assert num_shards > 1, "ERROR: num shards must be greater than 1 but is {}.".format(
            num_shards
        )
        self.num_shards = num_shards

        super(BatchSplitter, self).__init__(
            scope=kwargs.pop("scope", "batch-splitter"),
            graph_fn_num_outputs=dict(_graph_fn_split_batch=num_shards),
            **kwargs
        )
        self.define_api_method(name="split_batch", func=self._graph_fn_split_batch)

    def _graph_fn_split_batch(self, *inputs):
        """

        Args:
            *input (FlattenedDataOp): Input tensors which must all have the same batch dimension.

        Returns:
            List: List of DataOpTuples containing the input shards.
        """
        if get_backend() == "tf":
            batch_size = tf.shape(inputs[0])[0]
            shard_size = int(batch_size / self.num_shards)
            shards = list()

            # E.g. 203 elems in batch dim, 4 shards -> want 4 x 50
            usable_size = shard_size * batch_size
            for input_elem in inputs:
                # Must be evenly divisible so we slice out an evenly divisibl tensor.
                usable_input_tensor = input_elem[0:usable_size]
                shards.append(tf.split(value=usable_input_tensor, num_or_size_splits=self.num_shards, axis=0))

            # shards now has: 0th dim=input-arg; 1st dim=shards for this input-arg
            # The following is simply to flip the list so that it has:
            # 0th dim=shard number; 1st dim=input args
            flipped_list = list()
            for shard in range(self.num_shards):
                input_arg_list = list()
                for input_elem in range(len(inputs)):
                    input_arg_list.append(shards[input_elem][shard])
                flipped_list.append(DataOpTuple())
            return tuple(flipped_list)

