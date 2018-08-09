
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from rlgraph.components import Component
from rlgraph.utils import DataOpTuple, get_shape


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
        self.define_api_method(name="split", func=self._graph_fn_split_batch)

    def _graph_fn_split_batch(self, *inputs):
        """

        Args:
            *input (FlattenedDataOp): Input tensors which must all have the same batch dimension.

        Returns:
            List: List of DataOpTuples containing the input shards.
        """
        batch_size = get_shape(inputs[0])[0]
        shard_size = batch_size / self.num_shards
        ret = list()
        for input_elem in inputs:
            shard_list = []
            for i in range(self.num_shards):
                shard_list.append(input_elem[i * shard_size: i + 1 * shard_size])

            ret.append(DataOpTuple(shard_list))
        return ret
