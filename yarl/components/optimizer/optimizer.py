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


class OptimizerComponent(Component):
    """
    A component that takes a tuple of variables as in-Sockets and optimizes them according to some loss function
    or another criterion or method.
    """

    def __init__(self, scope="optimizer", *args, **kwargs):
        super(OptimizerComponent, self).__init__(scope=scope, *args, **kwargs)

        self.define_inputs("time_step", "variables")
        self.define_outputs("step")
        self.add_computation(["time_step", "variables"], "step", self._computation_step)

    def _computation_step(self, time_step, variables):
        """
        Performs a single optimization step on the incoming variables depending on this Component's setup and
        maybe the time-step

        Args:
            time_step (SingleDataOp): The time-step tensor.
            variables (DataOpTuple): The list of variables to be optimized.

        Returns:
            Op: A list of delta tensors corresponding to the updates for each optimized variable.
        """
        raise NotImplementedError

    def apply_step(self, variables, deltas):
        """
        Applies the given (and already calculated) step deltas to the variable values.

        Args:
            variables: List of variables.
            deltas: List of deltas of same length.

        Returns:
            The step-applied operation. A tf.group of tf.assign_add ops.
        """
        if len(variables) != len(deltas):
            raise YARLError("Invalid variables and deltas lists.")
        return tf.group(
            *(tf.assign_add(ref=variable, value=delta) for variable, delta in zip(variables, deltas))
        )
