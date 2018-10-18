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
from rlgraph.components import Component
from rlgraph.utils.decorators import rlgraph_api

if get_backend() == "tf":
    import tensorflow as tf
elif get_backend() == "pytorch":
    import torch


class LossFunction(Component):
    """
    A loss function component offers a simple interface into some error/loss calculation function.
    """
    def __init__(self, discount=0.98, **kwargs):
        """
        Args:
            discount (float): The discount factor (gamma).
        """
        super(LossFunction, self).__init__(scope=kwargs.pop("scope", "loss-function"), **kwargs)

        self.discount = discount

    @rlgraph_api
    def loss(self, *inputs):
        """
        API-method that calculates the total loss (average over per-batch-item loss) from the original input to
        per-item-loss.

        Args:
            see `self._graph_fn_loss_per_item`.

        Returns:
            Tuple (2x SingleDataOp):
                - The tensor specifying the final loss (over the entire batch).
                - The loss values vector (one single value for each batch item).
        """
        raise NotImplementedError

    @rlgraph_api
    def _graph_fn_loss_per_item(self, *inputs):
        """
        Returns the single loss values (one for each item in a batch).

        Args:
            *inputs (DataOpTuple): The various data that this function needs to calculate the loss.

        Returns:
            SingleDataOp: The tensor specifying the loss per item. The batch dimension of this tensor corresponds
                to the number of items in the batch.
        """
        raise NotImplementedError

    @rlgraph_api(must_be_complete=False)
    def _graph_fn_loss_average(self, loss_per_item):
        """
        The actual loss function that an optimizer will try to minimize. This is usually the average over a batch.

        Args:
            loss_per_item (SingleDataOp): The output of our loss_per_item graph_fn.

        Returns:
            SingleDataOp: The final loss tensor holding the average loss over the entire batch.
        """
        if get_backend() == "tf":
            return tf.reduce_mean(input_tensor=loss_per_item, axis=0)
        elif get_backend() == "pytorch":
            return torch.mean(loss_per_item, 0)
