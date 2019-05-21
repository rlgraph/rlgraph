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
from rlgraph.components.common.time_dependent_parameters import TimeDependentParameter
from rlgraph.components.loss_functions.dqn_loss_function import DQNLossFunction
from rlgraph.utils.decorators import rlgraph_api, graph_fn
from rlgraph.utils.util import get_rank

if get_backend() == "tf":
    import tensorflow as tf


class DQFDLossFunction(DQNLossFunction):
    """
    The DQFD-loss extends the (dueling) DQN loss by a supervised loss to leverage expert demonstrations. Paper:

    https://arxiv.org/abs/1704.03732

    API:
        loss_per_item(q_values_s, actions, rewards, terminals, qt_values_sp, q_values_sp=None): The DQN loss per batch
            item.
    """
    def __init__(self,  supervised_weight=1.0, scope="dqfd-loss-function", **kwargs):
        """
        Args:
            supervised_weight (spec(TimeDependentParameter)): Indicates weight of the expert loss.
        """
        super(DQFDLossFunction, self).__init__(scope=scope, **kwargs)

        self.supervised_weight = TimeDependentParameter.from_spec(supervised_weight)

        self.add_components(self.supervised_weight)

    @rlgraph_api
    def loss(self, q_values_s, actions, rewards, terminals, qt_values_sp, expert_margins,
             q_values_sp=None, importance_weights=None, apply_demo_loss=False):
        loss_per_item = self.loss_per_item(
            q_values_s, actions, rewards, terminals, qt_values_sp, expert_margins, q_values_sp,
            importance_weights, apply_demo_loss
        )
        total_loss = self.loss_average(loss_per_item)
        return total_loss, loss_per_item

    @rlgraph_api
    def loss_per_item(self, q_values_s, actions, rewards, terminals, qt_values_sp, expert_margins, q_values_sp=None,
                      importance_weights=None, apply_demo_loss=False, time_percentage=None):
        # Get the targets per action.
        td_targets = self._graph_fn_get_td_targets(rewards, terminals, qt_values_sp, q_values_sp)
        # Average over container sub-actions.
        if self.shared_container_action_target is True:
            td_targets = self._graph_fn_average_over_container_keys(td_targets)

        # Calculate the loss per item.
        loss_per_item = self._graph_fn_loss_per_item(td_targets, q_values_s, actions, expert_margins,
                                                     importance_weights, apply_demo_loss, time_percentage)
        # Average over container sub-actions.
        loss_per_item = self._graph_fn_average_over_container_keys(loss_per_item)

        # Apply huber loss.
        loss_per_item = self._graph_fn_apply_huber_loss_if_necessary(loss_per_item)

        return loss_per_item

    @graph_fn(flatten_ops=True, split_ops=True, add_auto_key_as_first_param=True)
    def _graph_fn_loss_per_item(self, key, td_targets, q_values_s, actions, expert_margins,
                                importance_weights=None, apply_demo_loss=False, time_percentage=None):
        """
        Args:
            td_targets (SingleDataOp): The already calculated TD-target terms (r + gamma maxa'Qt(s',a')
                OR for double Q: r + gamma Qt(s',argmaxa'(Q(s',a'))))
            q_values_s (SingleDataOp): The batch of Q-values representing the expected accumulated discounted returns
                when in s and taking different actions a.
            actions (SingleDataOp): The batch of actions that were actually taken in states s (from a memory).
            importance_weights (Optional[SingleDataOp]): If 'self.importance_weights' is True: The batch of weights to
                apply to the losses.
            apply_demo_loss (Optional[SingleDataOp]): If 'apply_demo_loss' is True: The large-margin loss is applied.
                Should be set to True when updating from demo data, False when updating from online data.
            expert_margins (SingleDataOp): The expert margin enforces a distance in Q-values between expert action and
                all other actions.
        Returns:
            SingleDataOp: The loss values vector (one single value for each batch item).
        """
        if get_backend() == "tf":
            # Q(s,a) -> Use the Q-value of the action actually taken before.
            one_hot = tf.one_hot(indices=actions, depth=self.flat_action_space[key].num_categories)
            q_s_a_values = tf.reduce_sum(input_tensor=(q_values_s * one_hot), axis=-1)

            # Calculate the TD-delta (targets - current estimate).
            td_delta = td_targets - q_s_a_values

            # Calculate the demo-loss.
            #  J_E(Q) = max_a([Q(s, a_taken) + l(s, a_expert, a_taken)] - Q(s, a_expert)
            mask = tf.ones_like(tensor=one_hot, dtype=tf.float32)
            action_mask = mask - one_hot

            # Margin mask: allow custom per-sample expert margins -> requires creating a margin matrix.
            # Instead of applying the same margin to all samples, users can pass a margin vector.
            # Broadcast to one hot shape
            expert_margins = tf.expand_dims(expert_margins, -1)
            expert_margins = tf.broadcast_to(input=expert_margins, shape=tf.shape(one_hot))
            margin_mask = expert_margins - one_hot

            # margin_mask = tf.Print(margin_mask, [margin_mask], summarize=100, message="margin mask =")
            margin_val = action_mask * margin_mask
            loss_input = q_values_s + margin_val

            # Apply margin.
            def map_margins(x):
                element_margin = x[0]
                element_loss = x[1]
                # Positive margins: apply max.
                # Negative margins: apply min.
                return tf.cond(
                    pred=tf.reduce_sum(element_margin) > 0,
                    true_fn=lambda: tf.reduce_max(element_loss),
                    false_fn=lambda: tf.reduce_min(element_loss),
                )
            supervised_loss = tf.map_fn(map_margins, (margin_val, loss_input), dtype=tf.float32)

            # Subtract Q-values of action actually taken.
            supervised_delta = supervised_loss - q_s_a_values
            td_delta = tf.cond(
                pred=apply_demo_loss,
                true_fn=lambda: td_delta + self.supervised_weight.get(time_percentage) * supervised_delta,
                false_fn=lambda: td_delta
            )

            # Reduce over the composite actions, if any.
            if get_rank(td_delta) > 1:
                td_delta = tf.reduce_mean(input_tensor=td_delta, axis=list(range(1, self.ranks_to_reduce + 1)))

            # Apply importance-weights from a prioritized replay to the loss.
            if self.importance_weights:
                return importance_weights * td_delta
            else:
                return td_delta
