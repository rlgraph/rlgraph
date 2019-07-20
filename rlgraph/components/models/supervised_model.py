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
    def __init__(self, supervised_predictor_spec, loss_function_spec, optimizer_spec,
                 scope="supervised-model", **kwargs):
        """
        Args:
            supervised_predictor (SupervisedPredictor): The SupervisedPredictor Component to use.
            loss_function (LossFunction): The loss function Component to use.
            optimizer (Optimizer): The optimizer Component to use.
        """
        super(SupervisedModel, self).__init__(scope=scope, **kwargs)

        self.supervised_predictor = SupervisedPredictor.from_spec(supervised_predictor_spec)
        self.loss_function = LossFunction.from_spec(loss_function_spec)
        self.optimizer = Optimizer.from_spec(optimizer_spec)

        self.add_components(self.supervised_predictor, self.loss_function, self.optimizer)

    @rlgraph_api
    def predict(self, nn_inputs, deterministic=None):
        return self.supervised_predictor.predict(nn_inputs, deterministic=deterministic)

    @rlgraph_api
    def get_distribution_parameters(self, nn_inputs):
        return self.supervised_predictor.get_distribution_parameters(nn_inputs)

    @rlgraph_api
    def update(self, nn_inputs, labels, timestep=None):
        parameters = self.supervised_predictor.get_distribution_parameters(nn_inputs)
        loss, loss_per_item = self.loss_function.loss(parameters, labels, timestep)
        step_op, _, _ = self.optimizer.step(self.supervised_predictor.variables(), loss, loss_per_item)

        return dict(
            step_op=step_op,
            loss=loss,
            loss_per_item=loss_per_item,
            parameters=parameters
        )
