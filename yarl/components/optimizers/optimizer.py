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
        variables (tuple): The tuple of (trainable) variables to be optimized.
        *inputs (any): Other necessary inputs for the specific type of optimizer (e.g. a time-step).
    outs:
        deltas (tuple): The tuple of delta tensors to be added to each of the variables in the `variables` in-Socket.
        step (tuple): Same as `deltas`, but also triggers actually applying the deltas to the `variables`.
    """
    def __init__(self, learning_rate, loss_function, *inputs, **kwargs):
        """
        Args:
            learning_rate (float): The learning rate to use.
            loss_function (Component): The LossFunction (Component) to minimize.
        """
        super(Optimizer, self).__init__(scope=kwargs.pop("scope", "optimizer"), **kwargs)

        self.learning_rate = learning_rate
        self.loss_function = loss_function

        # Define our interface.
        self.define_inputs("variables", *inputs)
        self.define_outputs("deltas", "step")
        self.add_graph_fn(["variables"] + inputs, "deltas", self._graph_fn_calculate_deltas)
        self.add_graph_fn(["variables", "deltas"], "step", self._graph_fn_apply_deltas)

    def _graph_fn_calculate_deltas(self, *inputs):
        """
        Performs a single optimization step on the incoming variables depending on this Component's setup and
        maybe the time-step

        Args:
            time_step (SingleDataOp): The time-step tensor.
            variables (DataOpTuple): The list of variables to be optimized.

        Returns:
            DataOpTuple: A list of delta tensors corresponding to the updates for each optimized variable.
        """
        raise NotImplementedError

    def _graph_fn_apply_deltas(self, variables, deltas):
        """
        Performs a single optimization step on the incoming variables depending on this Component's setup and
        maybe the time-step

        Args:
            time_step (SingleDataOp): The time-step tensor.
            variables (DataOpTuple): The list of variables to be optimized.

        Returns:
            DataOpTuple: A list of delta tensors corresponding to the updates for each optimized variable.
        """
        raise NotImplementedError
