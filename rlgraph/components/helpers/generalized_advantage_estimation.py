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
from rlgraph.components.helpers import SequenceHelper
from rlgraph.utils.decorators import rlgraph_api

if get_backend() == "tf":
    import tensorflow as tf


class GeneralizedAdvantageEstimation(Component):
    """
    A Helper Component that contains a graph_fn to generalized advantage estimation (GAE) [1].

    [1] High-Dimensional Continuous Control Using Generalized Advantage Estimation - Schulman et al.
    - 2015 (https://arxiv.org/abs/1506.02438)
    """

    def __init__(self, gae_lambda=1.0, discount=1.0, device="/device:CPU:0",
                 scope="generalized-advantage-estimation", **kwargs):
        """
        Args:
            gae_lambda (float): GAE-lambda. See paper for details.
            discount (float): Discount gamma.
        """
        super(GeneralizedAdvantageEstimation, self).__init__(device=device, scope=scope, **kwargs)
        self.gae_lambda = gae_lambda
        self.discount = discount
        self.sequence_helper = SequenceHelper()
        self.add_components(self.sequence_helper)

    @rlgraph_api(must_be_complete=False, returns=2)
    def _graph_fn_calc_gae_values(self, baseline_values, rewards, terminals):
        """
        Returns advantage values based on GAE.

        Args:
            baseline_values (DataOp): Baseline predictions V(s).
            rewards (DataOp): Rewards in sample trajectory.
            terminals (DataOp): Terminals in sample trajectory.

        Returns:
            PG-advantage values used for training via policy gradient with baseline.
        """
        if get_backend() == "tf":
            num_values = tf.shape(baseline_values)[0]
            gae_discount = self.gae_lambda * self.discount
            # Use helper to calculate sequence lengths and decay sequences.

            # Next, we need to set the next value after the end of each subsequence to 0/its prior value
            # depending on terminal.
            bootstrap_value = baseline_values[-1]
            adjusted_values = tf.TensorArray(dtype=tf.float32, infer_shape=False,
                                             size=1, dynamic_size=True, clear_after_read=False)

            def write(write_index, values, value):
                values = values.write(write_index, value)
                write_index += 1
                return values, write_index

            def body(index, write_index, values):
                values = values.write(write_index, baseline_values[index])
                write_index += 1

                # Append 0 whenever we terminate.
                values, write_index = tf.cond(
                    pred=tf.equal(terminals[index], 1),
                    true_fn=lambda: write(write_index, values, 0.0),
                    false_fn=lambda: (values, write_index)
                )
                return index + 1, write_index, values

            def cond(index, write_index, values):
                return index < num_values

            index, write_index, adjusted_values = tf.while_loop(
                cond=cond,
                body=body,
                loop_vars=[0, 0, adjusted_values],
                back_prop=False
            )

            # In case the last element was not a terminal, append boot_strap_value.
            # If was terminal -> already appended in loop.
            values, _ = tf.cond(pred=tf.greater(terminals[-1], 0),
                                true_fn=lambda: (adjusted_values, write_index),
                                false_fn=lambda: write(write_index, adjusted_values, bootstrap_value))

            adjusted_v = values.stack()
            rewards = tf.Print(rewards, [rewards], message="rewards = ")
            adjusted_v = tf.Print(adjusted_v, [adjusted_v], message="adjusted_v = ")
            deltas = rewards + self.discount * adjusted_v[1:] - adjusted_v[:-1]

            # Apply gae discount to each sub-sequence.
            advantages = self.sequence_helper.reverse_apply_decays_to_sequence(deltas, terminals, gae_discount)
            return advantages
