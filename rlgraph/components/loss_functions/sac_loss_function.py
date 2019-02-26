# Copyright 2018/2019 The RLgraph authors. All Rights Reserved.
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
from rlgraph.components.loss_functions.loss_function import LossFunction
from rlgraph.spaces.space_utils import sanity_check_space
from rlgraph.utils.decorators import rlgraph_api, graph_fn

if get_backend() == "tf":
    import tensorflow as tf


class SACLossFunction(LossFunction):
    """
    TODO: docs
    """
    def __init__(self, target_entropy=None, discount=0.99, scope="sac-loss-function", **kwargs):
        super(SACLossFunction, self).__init__(discount=discount, scope=scope, **kwargs)
        self.target_entropy = target_entropy

    def check_input_spaces(self, input_spaces, action_space=None):
        # All the following need a batch rank.
        for in_space_name in ["log_probs_sampled", "log_probs_next_sampled", "q_values", "q_values_sampled",
                              "q_values_next_sampled", "rewards", "terminals"]:
            in_space = input_spaces[in_space_name]
            sanity_check_space(in_space, must_have_batch_rank=True, must_have_time_rank=False)

        # All the following need shape==().
        for in_space_name in ["alpha", "rewards", "terminals"]:
            in_space = input_spaces[in_space_name]
            sanity_check_space(in_space, shape=())

        # All the following need shape==(1,).
        for in_space_name in ["log_probs_sampled", "log_probs_next_sampled", "q_values",
                              "q_values_sampled", "q_values_next_sampled"]:
            in_space = input_spaces[in_space_name]
            sanity_check_space(in_space, shape=(1,))

    @rlgraph_api
    def loss(self, alpha, log_probs_next_sampled, q_values_next_sampled, q_values, log_probs_sampled,
             q_values_sampled, rewards, terminals):
        actor_loss_per_item, critic_loss_per_item, alpha_loss_per_item = self.loss_per_item(
            alpha, log_probs_next_sampled, q_values_next_sampled, q_values, log_probs_sampled,
            q_values_sampled, rewards, terminals
        )
        actor_loss = self.loss_average(actor_loss_per_item)
        critic_loss = self.loss_average(critic_loss_per_item)
        alpha_loss = self.loss_average(alpha_loss_per_item)
        return actor_loss, actor_loss_per_item, critic_loss, critic_loss_per_item, alpha_loss, alpha_loss_per_item

    @graph_fn
    def _graph_fn__critic_loss(self, alpha, log_probs_next_sampled, q_values_next_sampled, q_values, rewards, terminals):
        q_min_next = tf.reduce_min(tf.concat(q_values_next_sampled, axis=1), axis=1, keepdims=True)
        assert q_min_next.shape.as_list() == [None, 1]
        soft_state_value = q_min_next - alpha * log_probs_next_sampled
        q_target = rewards + self.discount * (1.0 - tf.cast(terminals, tf.float32)) * soft_state_value
        total_loss = 0.0
        for i, q_value in enumerate(q_values):
            loss = 0.5 * (q_value - tf.stop_gradient(q_target)) ** 2
            loss = tf.identity(loss, "critic_loss_per_item_{}".format(i + 1))
            total_loss += loss
        return total_loss

    @graph_fn
    def _graph_fn__actor_loss(self, alpha, log_probs_sampled, q_values_sampled):
        q_min = tf.reduce_min(tf.concat(q_values_sampled, axis=1), axis=1, keepdims=True)
        assert q_min.shape.as_list() == [None, 1]
        loss = alpha * log_probs_sampled - q_min
        loss = tf.identity(loss, "actor_loss_per_item")
        return loss

    @graph_fn
    def _graph_fn__alpha_loss(self, alpha, log_probs_sampled):
        loss = -alpha * tf.stop_gradient(log_probs_sampled + self.target_entropy)
        loss = tf.identity(loss, "alpha_loss_per_item")
        return loss

    @rlgraph_api
    def _graph_fn_loss_per_item(self, alpha, log_probs_next_sampled, q_values_next_sampled, q_values, log_probs_sampled,
                                q_values_sampled, rewards, terminals):
        rewards = tf.expand_dims(rewards, axis=-1)
        terminals = tf.expand_dims(terminals, axis=-1)

        critic_loss_per_item = self._graph_fn__critic_loss(
            alpha=alpha,
            log_probs_next_sampled=log_probs_next_sampled,
            q_values_next_sampled=q_values_next_sampled,
            q_values=q_values,
            rewards=rewards,
            terminals=terminals
        )
        critic_loss_per_item = tf.squeeze(critic_loss_per_item, axis=1)

        actor_loss_per_item = self._graph_fn__actor_loss(
            alpha=alpha,
            log_probs_sampled=log_probs_sampled,
            q_values_sampled=q_values_sampled
        )
        actor_loss_per_item = tf.squeeze(actor_loss_per_item, axis=1)

        if self.target_entropy is not None:
            alpha_loss_per_item = self._graph_fn__alpha_loss(alpha=alpha, log_probs_sampled=log_probs_sampled)
            alpha_loss_per_item = tf.squeeze(alpha_loss_per_item, axis=1)
        else:
            # TODO: optimize this path
            alpha_loss_per_item = tf.zeros([tf.shape(rewards)[0]])

        return actor_loss_per_item, critic_loss_per_item, alpha_loss_per_item
