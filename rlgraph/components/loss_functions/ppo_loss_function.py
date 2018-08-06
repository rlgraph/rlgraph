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
from rlgraph.components import Categorical
from rlgraph.components.loss_functions import LossFunction
from rlgraph.spaces import IntBox, FloatBox
from rlgraph.spaces.space_utils import sanity_check_space

if get_backend() == "tf":
    import tensorflow as tf


class PPOLossFunction(LossFunction):
    """
    Loss function for proximal policy optimization:

    https://arxiv.org/abs/1707.06347
    """
    def __init__(self, clip_ratio=0.2, scope="ppo-loss-function", **kwargs):
        """
        Args:
            clip_ratio (float): How much to clip the likelihood ratio between old and new policy when updating.
            **kwargs:
        """
        self.clip_ratio = clip_ratio

        ## Pass our in-Socket names to parent constructor.
        #input_sockets = ["actions", "rewards", "terminals"]
        super(PPOLossFunction, self).__init__(scope=scope, **kwargs)

        self.action_space = None
        # How many ranks do we have to reduce to get down to the final loss per batch item?
        self.ranks_to_reduce = 0
        self.distribution = None

    def check_input_spaces(self, input_spaces, action_space=None):
        """
        Do some sanity checking on the incoming Spaces:
        """
        assert action_space is not None
        self.action_space = action_space
        # Check for IntBox and FloatBox.?
        sanity_check_space(
            self.action_space, allowed_types=[IntBox, FloatBox], must_have_categories=False
        )
        self.ranks_to_reduce = len(self.action_space.get_shape(with_batch_rank=True)) - 1

        # TODO: Make this flexible with different distributions.
        self.distribution = Categorical()

    def _graph_fn_loss_per_item(self, distribution, actions, rewards, terminals, prev_log_likelihood):
        """
        Args:
            distribution (Distribution): Distribution object which must provide a log likelihood function.
            actions (SingleDataOp): The batch of actions that were actually taken in states s (from a memory).
            rewards (SingleDataOp): The batch of rewards that we received after having taken a in s (from a memory).
            terminals (SingleDataOp): The batch of terminal signals that we received after having taken a in s
                (from a memory).
            prev_log_likelihood (SingleDataOp): Log likelihood to compare to when computing likelihood ratios.
        Returns:
            SingleDataOp: The loss values vector (one single value for each batch item).
        """
        if get_backend() == "tf":
            # Call graph_fn of a sub-Component directly.
            current_log_likelihood = self.distribution._graph_fn_log_prob(distribution, values=actions)

            likelihood_ratio = tf.exp(x=(current_log_likelihood / prev_log_likelihood))
            unclipped_objective = likelihood_ratio * rewards
            clipped_objective = tf.clip_by_value(
                t=likelihood_ratio,
                clip_value_min=(1 - self.clip_ratio),
                clip_value_max=(1 + self.clip_ratio),
            ) * rewards

            surrogate_objective = tf.minimum(x=unclipped_objective, y=clipped_objective)
            return tf.reduce_mean(input_tensor=-surrogate_objective, axis=0)

