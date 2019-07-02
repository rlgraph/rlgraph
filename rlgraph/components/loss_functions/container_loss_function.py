# Copyright 2018/2019 ducandu GmbH, All Rights Reserved.
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

from rlgraph.components.loss_functions.loss_function import LossFunction
from rlgraph.components.loss_functions.supervised_loss_function import SupervisedLossFunction
from rlgraph.utils.decorators import rlgraph_api


class ContainerLossFunction(SupervisedLossFunction):
    """
    A loss function consisting of n sub-loss functions whose weighted sum is used as the total loss.
    """
    def __init__(self, loss_functions_spec, weights=None, scope="mixture-loss", **kwargs):
        """
        Args:
            loss_functions_spec (Union[Dict[str,dict],Tuple[dict]]): A specification dict or tuple with values being
                the spec dicts for the single loss functions. The `loss` methods expect a dict input or a single
                tuple input (not as *args) in its first parameter.

            weights (Optional[List[float]]): If given, sum over all sub loss function will be weighted.
        """
        super(ContainerLossFunction, self).__init__(scope=scope, **kwargs)

        # Create all component loss functions and store each one's weight.
        if isinstance(loss_functions_spec, dict):
            weights_ = {}
            self.loss_functions = {}
            for i, (key, loss_fn_spec) in enumerate(loss_functions_spec.items()):
                if weights is None and "weight" in loss_fn_spec:
                    weights_[key] = loss_fn_spec.pop("weight")
                # Change scope.
                if isinstance(loss_fn_spec, LossFunction):
                    loss_fn_spec.scope = loss_fn_spec.global_scope = loss_fn_spec.name = "loss-function-{}".format(i)
                    loss_fn_spec.propagate_scope(None)
                self.loss_functions[key] = LossFunction.from_spec(loss_fn_spec, scope="loss-function-{}".format(i))
        else:
            assert isinstance(loss_functions_spec, (list, tuple)),\
                "ERROR: `loss_functions_spec` must be dict or tuple/list!"
            weights_ = []
            self.loss_functions = []
            for i, loss_fn_spec in enumerate(loss_functions_spec):
                if weights is None and "weight" in loss_fn_spec:
                    weights_.append(loss_fn_spec.pop("weight"))
                # Change scope.
                if isinstance(loss_fn_spec, LossFunction):
                    loss_fn_spec.scope = loss_fn_spec.global_scope = loss_fn_spec.name = "loss-function-{}".format(i)
                    loss_fn_spec.propagate_scope(None)
                self.loss_functions.append(LossFunction.from_spec(loss_fn_spec, scope="loss-function-{}".format(i)))

        # Weights were given per component loss? If no weights given at all, use 1.0 for all loss components.
        if weights is None and len(weights_) > 0:
            weights = weights_
        self.weights = weights

        # Add all sub-Components.
        self.add_components(
            *list(self.loss_functions.values() if isinstance(loss_functions_spec, dict) else self.loss_functions)
        )

    @rlgraph_api
    def _graph_fn_loss_per_item(self, parameters, labels, sequence_length=None, time_percentage=None):
        """
        Args:
            predictions (ContainerDataOp): The container parameters, each one represents the input for one of our sub
                loss functions.

            labels (ContainerDataOp): The container labels.
            sequence_length: The lengths of each sequence (if applicable) in the given batch.
        """
        weighted_sum_loss_per_item = None
        # Feed all inputs through their respective loss function and do the weighted sum.
        if isinstance(self.loss_functions, dict):
            for key, loss_fn in self.loss_functions.items():
                loss_per_item = loss_fn.loss_per_item(parameters[key], labels[key], sequence_length, time_percentage)
                if self.weights is not None:
                    loss_per_item *= self.weights[key]
                if weighted_sum_loss_per_item is None:
                    weighted_sum_loss_per_item = loss_per_item
                else:
                    weighted_sum_loss_per_item += loss_per_item
        else:
            for i, loss_fn in enumerate(self.loss_functions):
                loss_per_item = loss_fn.loss_per_item(parameters[i], labels[i], sequence_length, time_percentage)
                if self.weights is not None:
                    loss_per_item *= self.weights[i]
                if weighted_sum_loss_per_item is None:
                    weighted_sum_loss_per_item = loss_per_item
                else:
                    weighted_sum_loss_per_item += loss_per_item

        return weighted_sum_loss_per_item
