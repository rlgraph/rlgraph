
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from rlgraph import get_backend
from rlgraph.components.component import Component
from rlgraph.components.layers.preprocessing.reshape import ReShape
from rlgraph.utils.decorators import rlgraph_api, graph_fn


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
    def apply(self, input_):
        folded = self._graph_fn_fold(input_)
        applied = self._graph_fn_apply(folded)
        unfolded = self._graph_fn_unfold(applied, input_)
        return unfolded

    @graph_fn(flatten_ops=True, split_ops=True)
    def _graph_fn_fold(self, input_):
        if get_backend() == "tf":
            # Fold the time rank.
            input_folded = self.folder.apply(input_)
            return input_folded

    @graph_fn
    def _graph_fn_apply(self, input_folded):
        if get_backend() == "tf":
            # Send the folded input through the sub-component.
            sub_component_out = getattr(self.sub_component, self.api_method_name)(input_folded)
            return sub_component_out

    @graph_fn(flatten_ops=True, split_ops=True)
    def _graph_fn_unfold(self, sub_component_out, orig_input):
        if get_backend() == "tf":
            # Un-fold the time rank again.
            output = self.unfolder.apply(sub_component_out, input_before_time_rank_folding=orig_input)
            return output

