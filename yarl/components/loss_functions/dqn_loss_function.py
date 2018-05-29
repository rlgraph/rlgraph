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

from yarl import backend
from yarl.components.loss_functions import LossFunction


class DQNLossFunction(LossFunction):
    """
    The classic 2015 DQN Loss Function:
    L = Expectation-over-uniform-batch(r + gamma x maxa'Qt(s',a') - Qn(s,a))²
    Where Qn is the "normal" Q-network and Qt is the "target" net (that's a little behind Qn for stability purposes).
    """
    def __init__(self, discount=0.98, scope="dqn-loss-function", **kwargs):
        super(DQNLossFunction, self).__init__(scope=scope, **kwargs)
        self.discount = discount

    def _computation_loss_per_item(self, q_values, actions, rewards, q_values_s_):
        """
        Args:
            q_values (DataOp): The Q-values representing the expected accumulated discounted return when in s and
                taking different actions a.
            actions (DataOp): The actions that were actually taken in states s.
            rewards (DataOp): The rewards that we received after having taken a in s.
            q_values_s_ (DataOp): The Q-values representing the expected accumulated discounted return when in s'
                and taking different actions a'.

        Returns:
            DataOp: The average loss value of the batch.
        """
        if backend() == "tf":
            import tensorflow as tf
            # Construct gather indexes from actions.
            gather_indexes = []  # TODO:
            td_delta = (rewards + self.discount * tf.reduce_max(q_values_s_, axis=-1)) -\
                       tf.gather_nd(q_values, gather_indexes)
            return tf.pow(td_delta, 2)

