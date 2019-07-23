# Copyright 2018/2019 The RLgraph authors. All Rights Reserved.
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

from rlgraph import get_backend
from rlgraph.components.models.supervised_model import SupervisedModel
from rlgraph.components.optimizers.optimizer import Optimizer
from rlgraph.components.component import Component
from rlgraph.utils.decorators import rlgraph_api, graph_fn

if get_backend() == "tf":
    import tensorflow as tf


class SupervisedRootComponent(Component):
    """
    The root component for a SupervisedLearner.
    """
    def __init__(self, learner, *, supervised_model_spec, scope="supervised-root-component", **kwargs):
        """
        Args:
            learner (Optional[Learner]): The associated Learner object.

            model_spec (dict): The Model Component spec to be used for supervised learning.
        """

        super(SupervisedRootComponent, self).__init__(scope=scope, **kwargs)

        # Our owning Learner object (may be None, e.g. for testing purposes).
        self.learner = learner

        # Root component, set nesting level to 0.
        self.nesting_level = 0

        self.supervised_model = SupervisedModel.from_spec(supervised_model_spec)
        self.add_components(self.supervised_model)

        self.all_optimizers = self.supervised_model.optimizer

    def add_components(self, *sub_components, expose_apis=None):
        super(SupervisedRootComponent, self).add_components(*sub_components, expose_apis=expose_apis)

        # Keep track of all our Optimizers.
        for sub_component in self.get_all_sub_components(exclude_self=True):
            if isinstance(sub_component, Optimizer):
                self.all_optimizers.append(sub_component)

    @rlgraph_api
    def predict(self, prediction_input):
        out = self.supervised_model.predict(prediction_input)
        return out["output"]

    @rlgraph_api
    def get_distribution_parameters(self, prediction_input):
        out = self.supervised_model.get_distribution_parameters(prediction_input)
        return out["parameters"]

    @rlgraph_api
    def get_loss(self, prediction_input, labels, sequence_length=None):
        distribution_parameters = self.supervised_model.get_distribution_parameters(prediction_input)["parameters"]
        loss, loss_per_item = self.supervised_model.loss_function.loss(
            distribution_parameters, labels, sequence_length=sequence_length
        )
        return dict(loss=loss, loss_per_item=loss_per_item, distribution_parameters=distribution_parameters)

    @rlgraph_api
    def update(self, prediction_input, labels, learning_rate=None):
        #predictor = root.get_sub_component_by_name(learner.predictor.scope)
        #optimizer = root.get_sub_component_by_name(learner.optimizer.scope)
        sequence_length = None if learner.sequence_length_in_states is None else prediction_input[learner.sequence_length_in_states]
        out = self.get_loss(prediction_input, labels, sequence_length=sequence_length)

        predictor_vars = self.supervised_model.variables()
        step_op, loss, loss_per_item = self.supervised_model.optimizer.step(
            predictor_vars, out["loss"], out["loss_per_item"], learning_rate=learning_rate
        )
        return step_op, loss, loss_per_item

    @graph_fn
    def _graph_fn_training_step(self, other_step_op=None):
        """
        Increases the global training timestep by 1. Should be called by all training API-methods to
        timestamp each training/update step.

        Args:
            other_step_op (Optional[DataOp]): Another DataOp (e.g. a step_op) which should be
                executed before the increase takes place.

        Returns:
            DataOp: no_op.
        """
        if get_backend() == "tf":
            add_op = tf.assign_add(self.agent.graph_executor.global_training_timestep, 1)
            op_list = [add_op] + [other_step_op] if other_step_op is not None else []
            with tf.control_dependencies(op_list):
                if other_step_op is None or hasattr(other_step_op, "type") and other_step_op.type == "NoOp":
                    return tf.no_op()
                else:
                    return tf.identity(other_step_op)
        elif get_backend == "pytorch":
            self.agent.graph_executor.global_training_timestep += 1
            return None
