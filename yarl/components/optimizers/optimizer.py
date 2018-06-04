# Copyright 2018 The YARL-Project, All Rights Reserved.
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

from yarl.components import Component


class Optimizer(Component):
    """
    A component that takes a tuple of variables as in-Sockets and optimizes them according to some loss function
    or another criterion or method.

    API:
    ins:
        *inputs (any): Other necessary inputs for the specific type of optimizer (e.g. a time-step).
    outs:
        variables (tuple): The tuple of (trainable) variables to be optimized.
        deltas (tuple): The tuple of delta tensors to be added to each of the variables in the `variables` in-Socket.
        step (DataOp): Same as `deltas`, but also triggers actually applying the deltas to the `variables`.
    """
    def __init__(self, learning_rate, loss_function, *inputs, **kwargs):
        """
        Args:
            learning_rate (float): The learning rate to use.
            loss_function (Component): The LossFunction (Component) to minimize.
        """
        super(Optimizer, self).__init__(
            scope=kwargs.pop("scope", "optimizer"),
            flatten_ops=kwargs.pop("flatten_ops", False),
            **kwargs
        )

        self.learning_rate = learning_rate

        # TODO Note that we do not use this because we do not use minimize()
        self.loss_function = loss_function

        # Define our interface.
        self.define_inputs("variables", "loss", "apply_grads_and_vars", *inputs)
        self.define_outputs("calc_grads_and_vars", "step")

        self.add_graph_fn(
            inputs=["variables", "loss"] + list(inputs),
            outputs="calc_grads_and_vars",
            method=self._graph_fn_calculate_gradients
        )
        self.add_graph_fn(["apply_grads_and_vars"], "step", self._graph_fn_apply_gradients)

    def _graph_fn_calculate_gradients(self, variables, loss, *inputs):
        """
        Calculates the gradients for the given variables and the loss function (and maybe other child-class
            specific input parameters).

        Args:
            loss (SingeDataOp): The total loss over a batch to be minimized.
            variables (DataOpTuple): A list of variables to calculate gradients for.
            inputs (SingleDataOp): Custom SingleDataOp parameters, dependent on the optimizer type.

        Returns:
            DataOpTuple: The gradients per variable (same order as in input parameter `variables`).
        """
        raise NotImplementedError

    def _graph_fn_apply_gradients(self, grads_and_vars):
        """
        Changes the given variables based on the previously calculated gradients. `gradients` is the output of
            `self._graph_fn_calculate_gradients`.

        Args:
            grads_and_vars (DataOpTuple): The list of gradients and variables to be optimized.

        Returns:
            DataOp: The op to trigger the gradient-application step.
        """
        raise NotImplementedError
