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
from yarl.utils.util import get_rank
from yarl.components.loss_functions import LossFunction
from yarl.spaces import IntBox, sanity_check_space

if get_backend() == "tf":
    import tensorflow as tf


class PPOLossFunction(LossFunction):
    """
    Loss function for proximal policy optimization:

    https://arxiv.org/abs/1707.06347
    """

    def __init__(self, scope="ppo-loss-function", **kwargs):
        """

        """

        # Pass our in-Socket names to parent c'tor.
        input_sockets = ["actions", "rewards", "terminals"]

        super(PPOLossFunction, self).__init__(
            *input_sockets, scope=scope, **kwargs
        )
        self.action_space = None
        self.ranks_to_reduce = 0  # How many ranks do we have to reduce to get down to the final loss per batch item?

    def check_input_spaces(self, input_spaces, action_space):
        """
        Do some sanity checking on the incoming Spaces:
        """
        self.action_space = action_space
        # Check for IntBox and num_categories.
        sanity_check_space(
            self.action_space, allowed_types=[IntBox], must_have_categories=True
        )
        self.ranks_to_reduce = len(self.action_space.get_shape(with_batch_rank=True)) - 1

    def _graph_fn_loss_per_item(self, actions, rewards, terminals):
        """
        Args:

            actions (SingleDataOp): The batch of actions that were actually taken in states s (from a memory).
            rewards (SingleDataOp): The batch of rewards that we received after having taken a in s (from a memory).
            terminals (SingleDataOp): The batch of terminal signals that we received after having taken a in s
                (from a memory).
        Returns:
            SingleDataOp: The loss values vector (one single value for each batch item).
        """
        if get_backend() == "tf":
            pass

