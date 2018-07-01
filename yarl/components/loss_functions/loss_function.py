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

from yarl import get_backend
from yarl.components import Component

if get_backend() == "tf":
    import tensorflow as tf


class LossFunction(Component):
    """
    A loss function component offers a simple interface into some error/loss calculation function.

    API:
    ins:
        various api_methods (depending on specific loss function type (the Agent))
    outs:
        loss (SingleDataOp): The average loss value (over all single items in a batch).
        loss_per_item (SingleDataOp): The loss value vector holding single loss values (one per item in a batch).
    """
    def __init__(self, *inputs, **kwargs):
        """
        Args:
            *inputs (str): The names of our in-Sockets.

        Keyword Args:
            discount (float): The discount factor (gamma).
        """
        self.discount = kwargs.pop("discount", 0.98)

        super(LossFunction, self).__init__(scope=kwargs.pop("scope", "loss-function"), **kwargs)

        # Build our interface with a flexible number of in-Sockets.
        self.inputs = inputs
        # self.define_inputs(*self.inputs)
        # self.define_outputs("loss", "loss_per_item")
        self.define_api_method(name="loss_per_item", func=self._graph_fn_loss_per_item)
        self.define_api_method(name="loss", func=self._graph_fn_loss)

    def _graph_fn_loss_per_item(self, *inputs):
        """
        Returns the single loss values (one for each item in a batch).

        Args:
            *inputs (DataOpTuple): The various api_methods that this function needs to calculate the loss.

        Returns:
            SingleDataOp: The tensor specifying the loss per item. The batch dimension of this tensor corresponds
                to the number of items in the batch.
        """
        raise NotImplementedError

    def _graph_fn_loss(self, loss_per_item):
        """
        The actual loss function that an optimizer will try to minimize. This is usually the average over a batch.

        Args:
            loss_per_item (SingleDataOp): The output of our loss_per_item graph_fn.

        Returns:
            SingleDataOp: The final loss tensor holding the average loss over the entire batch.
        """
        if get_backend() == "tf":
            return tf.reduce_mean(input_tensor=loss_per_item, axis=0)

