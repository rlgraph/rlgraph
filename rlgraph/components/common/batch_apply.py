
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from rlgraph.components.component import Component
from rlgraph.components.layers.preprocessing.reshape import ReShape
from rlgraph.utils.decorators import rlgraph_api


class BatchApply(Component):
    """
    Takes an input with batch and time ranks, then folds the time rank into the batch rank,
    calls a certain API of some arbitrary child component, and unfolds the time rank again.
    """
    def __init__(self, sub_component, api_method_name, scope="batch-apply", **kwargs):
        """
        Args:
            sub_component (Component): The sub-Component to apply the batch to.
            api_method_name (str): The name of the API-method to call on the sub-component.
        """
        super(BatchApply, self).__init__(scope=scope, **kwargs)

        self.sub_component = sub_component
        self.api_method_name = api_method_name

        # Create the necessary reshape components.
        self.folder = ReShape(fold_time_rank=True, scope="folder")
        self.unfolder = ReShape(unfold_time_rank=True, scope="unfolder")

        self.add_components(self.sub_component, self.folder, self.unfolder)

    @rlgraph_api
    def call(self, input_):
        folded = self.folder.call(input_)
        applied = getattr(self.sub_component, self.api_method_name)(folded)
        unfolded = self.unfolder.call(applied, input_before_time_rank_folding=input_)
        return unfolded
