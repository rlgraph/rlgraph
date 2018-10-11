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

from rlgraph.components import Component
from rlgraph.utils.decorators import rlgraph_api


class Optimizer(Component):
    """
    A component that takes a tuple of variables as in-Sockets and optimizes them according to some loss function
    or another criterion or method.
    """
    def __init__(self, learning_rate=None, **kwargs):
        """
        Args:
            learning_rate (Optional[float]): The learning rate to use.
        """
        super(Optimizer, self).__init__(scope=kwargs.pop("scope", "optimizer"), **kwargs)

        self.learning_rate = learning_rate

    # Make all API-methods `must_be_complete`=False as optimizers don't implement `create_variables`.
    @rlgraph_api(must_be_complete=False)
    def _graph_fn_step(self, *inputs):
        """
        Applies an optimization step to a list of variables via a loss.

        Args:
            \*inputs (SingleDataOp): Any args to the optimizer to be able to perform gradient calculations from
                losses and then apply these gradients to some variables.

        Returns:

        """
        raise NotImplementedError

    @rlgraph_api(must_be_complete=False)
    def _graph_fn_calculate_gradients(self, *inputs):
        """
        Calculates the gradients for the given variables and the loss function (and maybe other child-class
        specific input parameters).

        Args:
            \*inputs (SingleDataOp): Custom SingleDataOp parameters, dependent on the optimizer type.

        Returns:
            DataOpTuple: The list of gradients and variables to be optimized.
        """
        raise NotImplementedError

    @rlgraph_api(must_be_complete=False)
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

    def get_optimizer_variables(self):
        """
        Returns this optimizer's variables. This extra utility function is necessary because
        some frameworks like TensorFlow create optimizer variables "late", e.g. Adam variables,
        so they cannot be fetched at graph build time yet.

        Returns:
            list: List of variables.
        """
        pass
