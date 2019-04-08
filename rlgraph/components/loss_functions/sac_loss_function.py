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
    def __init__(self, target_entropy=None, discount=0.99, num_q_functions=2, scope="sac-loss-function", **kwargs):
        super(SACLossFunction, self).__init__(discount=discount, scope=scope, **kwargs)
        self.num_q_functions = num_q_functions
        self.target_entropy = target_entropy

    def check_input_spaces(self, input_spaces, action_space=None):
        # All the following need a batch rank.
        self.action_space = action_space
        for in_space_name in ["log_probs_sampled", "log_probs_next_sampled", "q_values", "q_values_sampled",
                              "q_values_next_sampled", "rewards", "terminals"]:
            in_space = input_spaces[in_space_name]
            sanity_check_space(in_space, must_have_batch_rank=True, must_have_time_rank=False)

        # All the following need shape==().
        for in_space_name in ["alpha", "rewards", "terminals"]:
            in_space = input_spaces[in_space_name]
            sanity_check_space(in_space, shape=())

        # All the following need shape==(1,).
        for in_space_name in ["q_values", "q_values_sampled", "q_values_next_sampled"]:
            in_space = input_spaces[in_space_name]
            sanity_check_space(in_space, shape=(1,))

    @rlgraph_api
    def loss(self, alpha, log_probs_next_sampled, q_values_next_sampled, q_values, log_probs_sampled,
             q_values_sampled, rewards, terminals):
        actor_loss_per_item, critic_loss_per_item, alpha_loss_per_item = self.loss_per_item(
            alpha, log_probs_next_sampled, q_values_next_sampled, q_values, log_probs_sampled,
            q_values_sampled, rewards, terminals
        )

        # Average across batch.
        actor_loss = self.loss_average(actor_loss_per_item)
        critic_loss = self.loss_average(critic_loss_per_item)
        alpha_loss = self.loss_average(alpha_loss_per_item)
        return actor_loss, actor_loss_per_item, critic_loss, critic_loss_per_item, alpha_loss, alpha_loss_per_item

    @rlgraph_api
    def loss_per_item(self, alpha, log_probs_next_sampled, q_values_next_sampled, q_values, log_probs_sampled,
                      q_values_sampled, rewards, terminals):
        critic_loss_per_item = self._graph_fn_critic_loss(log_probs_next_sampled, q_values_next_sampled,
            q_values, rewards, terminals, alpha)
        critic_loss_per_item = self._graph_fn_average_over_container_keys(critic_loss_per_item)

        actor_loss_per_item = self._graph_fn_actor_loss(log_probs_sampled, q_values_sampled, alpha)
        actor_loss_per_item = self._graph_fn_average_over_container_keys(actor_loss_per_item)

        alpha_loss_per_item = self._graph_fn_alpha_loss(log_probs_sampled, alpha)
        alpha_loss_per_item = self._graph_fn_average_over_container_keys(alpha_loss_per_item)

        return actor_loss_per_item, critic_loss_per_item, alpha_loss_per_item

    @graph_fn(flatten_ops={0}, split_ops=True)
    def _graph_fn_critic_loss(self, log_probs_next_sampled, q_values_next_sampled, q_values, rewards, terminals, alpha):
        # In case log_probs come in as shape=(), expand last rank to 1.
        if log_probs_next_sampled.shape.as_list()[-1] is None:
            log_probs_next_sampled = tf.expand_dims(log_probs_next_sampled, axis=-1)

        log_probs_next_sampled = tf.reduce_sum(log_probs_next_sampled, axis=1, keepdims=True)
        rewards = tf.expand_dims(rewards, axis=-1)
        terminals = tf.expand_dims(terminals, axis=-1)

        q_min_next = tf.reduce_min(tf.concat(q_values_next_sampled, axis=1), axis=1, keepdims=True)
        assert q_min_next.shape.as_list() == [None, 1]
        soft_state_value = q_min_next - alpha * log_probs_next_sampled
        q_target = rewards + self.discount * (1.0 - tf.cast(terminals, tf.float32)) * soft_state_value
        total_loss = 0.0
        if self.num_q_functions < 2:
            q_values = [q_values]
        for i, q_value in enumerate(q_values):
            loss = 0.5 * (q_value - tf.stop_gradient(q_target)) ** 2
            loss = tf.identity(loss, "critic_loss_per_item_{}".format(i + 1))
            total_loss += loss
        return tf.squeeze(total_loss, axis=1)

    @graph_fn(flatten_ops={0}, split_ops=True)
    def _graph_fn_actor_loss(self, log_probs_sampled, q_values_sampled, alpha):
        if log_probs_sampled.shape.as_list()[-1] is None:
            log_probs_sampled = tf.expand_dims(log_probs_sampled, axis=-1)
        log_probs_sampled = tf.reduce_sum(log_probs_sampled, axis=1, keepdims=True)

        q_min = tf.reduce_min(tf.concat(q_values_sampled, axis=1), axis=1, keepdims=True)
        assert q_min.shape.as_list() == [None, 1]
        loss = alpha * log_probs_sampled - q_min
        loss = tf.identity(loss, "actor_loss_per_item")
        return tf.squeeze(loss, axis=1)

    @graph_fn(flatten_ops=True, split_ops=True)
    def _graph_fn_alpha_loss(self, log_probs_sampled, alpha):
        if self.target_entropy is None:
            return tf.zeros([tf.shape(log_probs_sampled)[0]])
        else:
            # in the paper this is -alpha * (log_pi + target entropy), however the implementation uses log_alpha
            # see the discussion in https://github.com/rail-berkeley/softlearning/issues/37
            loss = -tf.log(alpha) * tf.stop_gradient(log_probs_sampled + self.target_entropy)
            loss = tf.identity(loss, "alpha_loss_per_item")
            return tf.squeeze(loss)
