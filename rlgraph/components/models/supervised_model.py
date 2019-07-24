# Copyright 2018/2019 ducandu GmbH. All Rights Reserved.
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

from __future__ import absolute_import, division, print_function

from rlgraph.components.models.model import Model
from rlgraph.components.loss_functions.loss_function import LossFunction
from rlgraph.components.optimizers.optimizer import Optimizer
from rlgraph.components.policies.supervised_predictor import SupervisedPredictor
from rlgraph.utils.decorators import rlgraph_api


class SupervisedModel(Model):
    """
    A SupervisedModel is a Model that holds a SupervisedPredictor, a LossFunction and an Optimizer Component and
    exposes the Predictor's API plus some methods to update the SupervisedPredictor's parameters.
    """
    def __init__(self, supervised_predictor_spec, loss_function_spec, optimizer_spec, network_spec=None,
                 output_space=None, scope="supervised-model", **kwargs):
        """
        Args:
            supervised_predictor (Union[dict,SupervisedPredictor]): The SupervisedPredictor Component to use.
            loss_function_spec (Union[dict,LossFunction]): The loss function Component to use.
            optimizer_spec (Union[dict,Optimizer]): The optimizer Component to use.
        """
        super(SupervisedModel, self).__init__(scope=scope, **kwargs)

        assert supervised_predictor_spec is not None or (network_spec is not None and output_space is not None), \
            "ERROR: One of `supervised_predictor_spec` OR (`network_spec` and `output_space`) must be provided!"
        supervised_predictor_spec = supervised_predictor_spec or {
            "type": "supervised-predictor",
            "distribution_adapter_spec": {"output_space": self.output_space}
        }
        self.output_space = supervised_predictor_spec["output_space"]

        # Force the network spec on the predictor object.
        if network_spec is not None:
            supervised_predictor_spec["network_spec"] = network_spec
        # Force set output-space on predictor and its distribution adapter.
        if "output_space" not in supervised_predictor_spec:  # or "output_space" not in \
                #supervised_predictor_spec["distribution_adapter_spec"]:
            #supervised_predictor_spec["distribution_adapter_spec"]["output_space"]
            supervised_predictor_spec["output_space"] = self.output_space

        # Construct the wrapped Components.
        self.supervised_predictor = SupervisedPredictor.from_spec(supervised_predictor_spec)
        self.loss_function = LossFunction.from_spec(loss_function_spec)
        self.optimizer = Optimizer.from_spec(optimizer_spec)

        self.add_components(self.loss_function, self.optimizer)
        self.add_components(self.supervised_predictor)  #, expose_apis={"predict", "get_distribution_parameters"})

    @rlgraph_api
    def predict(self, nn_inputs, deterministic=None):
        return self.supervised_predictor.predict(nn_inputs, deterministic=deterministic)

    @rlgraph_api
    def get_distribution_parameters(self, nn_inputs):
        return self.supervised_predictor.get_distribution_parameters(nn_inputs)

    @rlgraph_api
    def update(self, nn_inputs, labels, time_percentage=None):
        parameters = self.supervised_predictor.get_distribution_parameters(nn_inputs)["parameters"]
        loss, loss_per_item = self.loss_function.loss(parameters, labels, time_percentage)
        step_op = self.optimizer.step(self.supervised_predictor.variables(), loss, loss_per_item, time_percentage)

        return dict(
            step_op=step_op,
            loss=loss,
            loss_per_item=loss_per_item,
            parameters=parameters
        )
