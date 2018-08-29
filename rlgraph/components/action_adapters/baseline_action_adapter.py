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
from rlgraph.components.action_adapters.action_adapter import ActionAdapter

if get_backend() == "tf":
    import tensorflow as tf


class BaselineActionAdapter(ActionAdapter):
    """
    An ActionAdapter that adds 1 node to its action layer for an additional state-value output per batch item.

    API:
        get_state_values_and_logits(nn_output) (Tuple[SingleDataOp x 2]): The state-value and action logits (reshaped).
    """
    def __init__(self, scope="baseline-action-adapter", **kwargs):
        # Change the number of units in the action layer (+1 for the extra Value function node).
        super(BaselineActionAdapter, self).__init__(add_units=1, scope=scope, **kwargs)

        self.input_space = None

    def check_input_spaces(self, input_spaces, action_space=None):
        self.input_space = input_spaces["nn_output"]

    def get_logits_parameters_log_probs(self, nn_output):
        """
        Override get_logits_parameters_log_probs API-method to not use the state-value, which must be sliced.
        """
        _, logits = self.call(self.get_state_values_and_logits, nn_output)
        return (logits,) + tuple(self.call(self._graph_fn_get_parameters_log_probs, logits))

    def get_state_values_and_logits(self, nn_output):
        """
        API-method. Returns separated V and logit values split from the action layer.

        Args:
            nn_output (DataOpRecord): The NN output of the preceding neural network.

        Returns:
            tuple (2x DataOpRecord):
                - The single state value (V).
                - The (reshaped) logit values for the different actions.
        """
        # Run through the action layer.
        action_layer_output = self.call(self.action_layer.apply, nn_output)
        # Slice away the first node for the state value and reshape the rest to yield the action logits.
        state_value, logits = self.call(self._graph_fn_get_state_values_and_logits, action_layer_output)
        return state_value, logits

    def _graph_fn_get_state_values_and_logits(self, action_layer_output):
        """
        Slices away the state-value node from the raw action_layer_output (dense) and returns the single state-value
        and the remaining (reshaped) action-logits.

        Args:
            action_layer_output (SingleDataOp): The flat action layer output.

        Returns:
            tuple (2x SingleDataOp):
                - The state value (as shape=(1,)).
                - The reshaped action logits.
        """
        if get_backend() == "tf":
            # Separate the single state-value node from the flat logits.
            state_value, flat_logits = tf.split(
                value=action_layer_output, num_or_size_splits=(1, self.action_layer.units - 1), axis=-1
            )

            # OBSOLETE: Don't squeeze
            ## Have to squeeze the state-value as it's coming from just one node anyway.
            #state_value = tf.squeeze(state_value, axis=-1)

            # TODO: automate this: batch in -> batch out; time in -> time out; batch+time in -> batch+time out, etc..
            # TODO: if not default behavior: have to specify in decorator (see design_problems.txt).
            # Now we have to reshape the flat logits to obtain the action-shaped logits.
            # Adjust batch/time ranks.
            flat_logits._batch_rank = 0 if self.input_space.time_major is False else 1
            if self.input_space.has_time_rank:
                flat_logits._time_rank = 0 if self.input_space.time_major is True else 1
            logits = self.call(self.reshape.apply, flat_logits)

            # TODO: automate this: batch in -> batch out; time in -> time out; batch+time in -> batch+time out, etc..
            # TODO: if not default behavior: have to specify in decorator (see design_problems.txt).
            # Adjust batch/time ranks.
            state_value._batch_rank = 0 if self.input_space.time_major is False else 1
            logits._batch_rank = 0 if self.input_space.time_major is False else 1
            if self.input_space.has_time_rank:
                state_value._time_rank = 0 if self.input_space.time_major is True else 1
                logits._time_rank = 0 if self.input_space.time_major is True else 1

            return state_value, logits
