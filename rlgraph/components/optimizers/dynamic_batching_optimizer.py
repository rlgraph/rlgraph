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

from rlgraph import get_backend
from rlgraph.components import LocalOptimizer, Component
from rlgraph.components.optimizers.optimizer import Optimizer
from rlgraph.utils.ops import DataOpTuple

if get_backend() == "tf":
    import tensorflow as tf


class DynamicBatchingOptimizer(Component):
    """
    A local optimizer performs optimization irrespective of any distributed semantics, i.e.
    it has no knowledge of other machines and does not implement any communications with them.
    """
    def __init__(self, optimizer_spec, **kwargs):
        super(DynamicBatchingOptimizer, self).__init__(scope=kwargs.pop("scope", "dynamic-batching-optimizer"), **kwargs)

        # The wrapped, backend-specific optimizer object.
        self.optimizer = LocalOptimizer.from_spec(optimizer_spec)

    def create_variables(self, input_spaces, action_space=None):
        # Must register the Optimizer's variables with the Component.
        # self.register_variables(*self.optimizer.variables())
        pass

    def _graph_fn_step(self, variables, loss, loss_per_item, *inputs):
        pass

    def _graph_fn_calculate_gradients(self, variables, loss):
        """
        Args:
            variables (DataOpTuple): A list of variables to calculate gradients for.
            loss (SingeDataOp): The total loss over a batch to be minimized.
        """
        return self.optimizer._graph_fn_calculate_gradients(variables, loss)

    def _graph_fn_apply_gradients(self, grads_and_vars):
        return self.optimizer._graph_fn_apply_gradients(grads_and_vars)

    def get_optimizer_variables(self):
        return self.optimizer.variables()
