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


class LossFunction(Component):
    """
    A loss function component offers a simple interface into some error/loss calculation function.

    API:
    ins:
        various inputs (depending on specific loss function type (the Agent))
    outs:
        loss (SingleDataOp): The average loss value (per single item in a batch).
    """
    def __init__(self, scope="loss-function", **kwargs):
        super(LossFunction, self).__init__(scope=scope, **kwargs)
        # Build our interface.
        # To be done by child classes.

    def _computation_loss(self, *inputs):
        """
        The actual loss function that an optimizer will try to minimize.

        Args:
            *inputs (DataOpTuple): The various inputs that this loss function needs to calculate the loss.

        Returns:
            SingleDataOp: The loss tensor.
        """
        raise NotImplementedError


