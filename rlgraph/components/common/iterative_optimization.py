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
from rlgraph.utils.decorators import graph_fn

from rlgraph.components import Component

if get_backend() == "tf":
    import tensorflow as tf


class IterativeOptimization(Component):
    """
    Sub-sampling optimizer loop.
    """
    def __init__(self, num_iterations, sample_size, optimizer, policy, loss, value_function,
                 vf_optimizer=None, scope="opt-loop", **kwargs):
        """
        Args:
            num_iterations (int): How often to call the optimizer step function.
            sample_size (int):
            optimizer (Optimizer): Optimizer
        """
        assert num_iterations > 0
        super(IterativeOptimization, self).__init__(scope=scope, **kwargs)

        self.num_iterations = num_iterations
        self.sample_size = sample_size

        self.optimizer = optimizer
        self.policy = policy
        self.loss = loss
        self.value_function = value_function
        self.vf_optimizer = vf_optimizer
        if vf_optimizer:
            self.add_components(self.vf_optimizer)
        self.add_components(optimizer)

    @graph_fn
    def _graph_fn_iterative_opt(self,  preprocessed_states, actions, rewards, terminals):
        """
        Calls iterative optimization by repeatedly subsampling.
        Returns:
            any: Result of the call.
        """

        if get_backend() == "tf":
            # Compute loss once to initialize loop.
            batch_size = tf.shape(preprocessed_states)[0]
            sample_indices = tf.random_uniform(shape=(self.sample_size,), maxval=batch_size, dtype=tf.int32)
            sample_states = tf.gather(params=preprocessed_states, indices=sample_indices)
            sample_actions = tf.gather(params=actions, indices=sample_indices)
            sample_rewards = tf.gather(params=rewards, indices=sample_indices)
            sample_terminals = tf.gather(params=terminals, indices=sample_indices)

            action_log_probs = self.policy.get_action_log_probs(sample_states, sample_actions)
            baseline_values = self.value_function.value_output(sample_states)

            loss, loss_per_item, vf_loss, vf_loss_per_item = self.loss.loss(
                action_log_probs, baseline_values, actions, sample_rewards, sample_terminals
            )

            # Args are passed in again because some device strategies may want to split them to different devices.
            policy_vars = self.policy.variables()
            vf_vars = self.value_function.variables()
            vf_step_op, vf_loss, vf_loss_per_item = self.vf_optimizer(vf_vars, vf_loss, vf_loss_per_item)
            step_op, loss, loss_per_item = self.optimizer.step(policy_vars, loss, loss_per_item)

            def opt_body(index, step_op, loss, loss_per_item, vf_step_op, vf_loss, vf_loss_per_item):
                with tf.control_dependencies([step_op, loss, loss_per_item, vf_step_op, vf_loss, vf_loss_per_item]):
                    batch_size = tf.shape(preprocessed_states)[0]
                    sample_indices = tf.random_uniform(shape=(self.sample_size,), maxval=batch_size, dtype=tf.int32)
                    sample_states = tf.gather(params=preprocessed_states, indices=sample_indices)
                    sample_actions = tf.gather(params=actions, indices=sample_indices)
                    sample_rewards = tf.gather(params=rewards, indices=sample_indices)
                    sample_terminals = tf.gather(params=terminals, indices=sample_indices)

                    action_log_probs = self.policy.get_action_log_probs(sample_states, sample_actions)
                    baseline_values = self.value_function.value_output(sample_states)

                    loss, loss_per_item, vf_loss, vf_loss_per_item = self.loss.loss(
                        action_log_probs, baseline_values, actions, sample_rewards, sample_terminals
                    )

                    # Args are passed in again because some device strategies may want to split them to different devices.
                    policy_vars = self.policy.variables()
                    vf_vars = self.value_function.variables()

                    vf_step_op, vf_loss, vf_loss_per_item = self.vf_optimizer(vf_vars, vf_loss, vf_loss_per_item)
                    step_op, loss, loss_per_item = self.optimizer.step(policy_vars, loss, loss_per_item)
                    return index, step_op, loss, loss_per_item, vf_step_op, vf_loss, vf_loss_per_item

            def cond(index, step_op, loss, loss_per_item, vf_step_op, v_loss, v_loss_per_item):
                return index < self.num_iterations

            index, step_op, loss, loss_per_item, vf_step_op, vf_loss, vf_loss_per_item = tf.while_loop(
                cond=cond,
                body=opt_body,
                # Start with 1.
                loop_vars=[1, step_op, loss, loss_per_item, vf_step_op, vf_loss, vf_loss_per_item],
                parallel_iterations=1
            )

            return step_op, loss, loss_per_item, vf_step_op, vf_loss, vf_loss_per_item
