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
from rlgraph.utils.decorators import rlgraph_api

if get_backend() == "tf":
    import tensorflow as tf
elif get_backend() == "pytorch":
    import torch


class SequenceHelper(Component):
    """
    A helper Component that helps manipulate sequences with various utilities, e.g. for discounting.
    """

    def __init__(self, scope="sequence-helper", **kwargs):
        super(SequenceHelper, self).__init__(scope=scope, **kwargs)

    @rlgraph_api(must_be_complete=False)
    def _graph_fn_calc_sequence_lengths(self, sequence_indices):
        """
        Computes sequence lengths for a tensor containing sequence indices, where 1 indicates start
        of a new sequence.
        Args:
            sequence_indices (DataOp): Indices denoting sequences, e.g. terminal values.
        Returns:
            Sequence lengths.
        """
        if get_backend() == "tf":
            # TensorArray:
            elems = tf.shape(input=sequence_indices)[0]
            sequence_lengths = tf.TensorArray(
                dtype=tf.int32,
                infer_shape=False,
                size=1,
                dynamic_size=True,
                clear_after_read=False
            )

            def update(write_index, sequence_array, length):
                # Write to index, increase
                sequence_array = sequence_array.write(write_index, length)
                return sequence_array, write_index + 1, 0

            def insert_body(index, length, sequence_lengths, write_index):
                length += 1

                # Update tensor array, reset length to 0.
                sequence_lengths, write_index, length = tf.cond(
                    pred=tf.equal(sequence_indices[index], 1),
                    true_fn=lambda: update(write_index, sequence_lengths, length),
                    false_fn=lambda: (sequence_lengths, write_index, length)
                )
                return index + 1, length, sequence_lengths, write_index

            def cond(index, length, sequence_lengths, write_index):
                return index < elems

            _, final_length, sequence_lengths, write_index = tf.while_loop(
                cond=cond,
                body=insert_body,
                loop_vars=[0, 0, sequence_lengths, 0],
                back_prop=False
            )
            # If the final element was terminal -> already included.
            sequence_lengths, _, _ = tf.cond(
                pred=tf.greater(final_length, 0),
                true_fn=lambda: update(write_index, sequence_lengths, final_length),
                false_fn=lambda: (sequence_lengths, write_index, final_length)
            )
            return sequence_lengths.stack()
        elif get_backend() == "pytorch":
            sequence_lengths = []
            length = 0
            for index in sequence_indices:
                length += 1
                if index == 1:
                    sequence_lengths.append(length)
                    length = 0
            # Append final sequence.
            if length > 0:
                sequence_lengths.append(length)
            return torch.tensor(sequence_lengths, dtype=torch.int32)

    @rlgraph_api(returns=2, must_be_complete=False)
    def _graph_fn_calc_sequence_decays(self, sequence_indices, decay):
        """
        Computes decays for sequence indices, e.g. for generalized advantage estimation.
        That is, a sequence with terminals is used to compute for each subsequence the decay
        values and the length of the sequence.

        Example:
        decay = 0.5, sequence_indices = [0 0 1 0 1] will return lengths [3, 2] and
        decays [1 0.5 0.25 1 0.5] (decay^0, decay^1, ..decay^k) where k = sequence length for
        each sub-sequence.

        Args:
            sequence_indices (DataOp): Indices denoting sequences, e.g. terminal values.
            decay (float): Initial decay value to start sub-sequence with.

        Returns:
            Sequence lengths and their decays.
        """
        if get_backend() == "tf":
            elems = tf.shape(input=sequence_indices)[0]
            # TensorArray:
            sequence_lengths = tf.TensorArray(
                dtype=tf.int32,
                infer_shape=False,
                size=1,
                dynamic_size=True,
                clear_after_read=False
            )
            decays = tf.TensorArray(
                dtype=tf.float32,
                infer_shape=False,
                size=1,
                dynamic_size=True,
                clear_after_read=False
            )

            def update(write_index, sequence_array, length):
                # Write to index, increase
                sequence_array = sequence_array.write(write_index, length)
                return sequence_array, write_index + 1, 0

            def insert_body(index, length, sequence_lengths, write_index, decays):
                # Decay is based on length, so val = decay^length
                decay_val = tf.pow(x=decay, y=tf.cast(length, dtype=tf.float32))

                # Write decay val into array.
                decays = decays.write(index, decay_val)
                length += 1

                # Update tensor array, reset length to 0.
                sequence_lengths, write_index, length = tf.cond(
                    pred=tf.equal(sequence_indices[index], 1),
                    true_fn=lambda: update(write_index, sequence_lengths, length),
                    false_fn=lambda: (sequence_lengths, write_index, length)
                )
                return index + 1, length, sequence_lengths, write_index, decays

            def cond(index, length, sequence_lengths, write_index, decays):
                return index < elems

            index, final_length, sequence_lengths, write_index, decays = tf.while_loop(
                cond=cond,
                body=insert_body,
                loop_vars=[0, 0, sequence_lengths, 0, decays],
                back_prop=False
            )

            # If the final element was terminal -> already included.
            # Decays need no updating because we just wrote them always.
            sequence_lengths, _, _ = tf.cond(
                pred=tf.greater(final_length, 0),
                true_fn=lambda: update(write_index, sequence_lengths, final_length),
                false_fn=lambda: (sequence_lengths, write_index, final_length)
            )
            return tf.stop_gradient(sequence_lengths.stack()), tf.stop_gradient(decays.stack())
        elif get_backend() == "pytorch":
            sequence_lengths = []
            decays = []

            length = 0
            for index in sequence_indices:
                # Compute decay based on sequence length.
                decays.append(pow(decay, length))
                length += 1
                if index == 1:
                    sequence_lengths.append(length)
                    length = 0

            # Append final sequence.
            if length > 0:
                sequence_lengths.append(length)
            return torch.tensor(sequence_lengths, dtype=torch.int32),\
                   torch.tensor(decays, dtype=torch.int32)

    @rlgraph_api(must_be_complete=False)
    def _graph_fn_reverse_apply_decays_to_sequence(self, values, sequence_indices, decay):
        """
        Computes decays for sequence indices and applies them (in reverse manner to a sequence of values).
        Useful to compute discounted reward estimates across a sequence of estimates.

        Args:
            values (DataOp): Values to apply decays to.
            sequence_indices (DataOp): Indices denoting sequences, e.g. terminal values.
            decay (float): Initial decay value to start sub-sequence with.

        Returns:
            Decayed sequence values.
        """
        if get_backend() == "tf":
            elems = tf.shape(input=sequence_indices)[0]
            decayed_values = tf.TensorArray(
                dtype=tf.float32,
                infer_shape=False,
                size=1,
                dynamic_size=True,
                clear_after_read=False
            )

            def insert_body(index, write_index, length, prev_v, decayed_values):
                # Reset length to 0 if terminal encountered.
                # NOTE: We cannot prev_v to 0.0 because values[index] might have a more complex shape,
                # so this violates shape checks.
                length, prev_v = tf.cond(
                    pred=tf.equal(sequence_indices[index], 1),
                    true_fn=lambda: (0, tf.zeros_like(prev_v)),
                    false_fn=lambda: (length, prev_v)
                )

                # Decay is based on length, so val = decay^length
                decay_val = tf.pow(x=decay, y=tf.cast(length, dtype=tf.float32))
                accum_v = prev_v + values[index] * decay_val

                # Write decayed val into array.
                decayed_values = decayed_values.write(write_index, accum_v)

                # Increase write-index and length of sub-sequence, decrease loop index in reverse iteration.
                return index - 1, write_index + 1, length + 1, accum_v, decayed_values

            def cond(index, write_index, length, prev_v, decayed_values):
                # Scan in reverse.
                return index >= 0

            _, _, _, _, decayed_values = tf.while_loop(
                cond=cond,
                body=insert_body,
                # loop index, index writing to tensor array, current length of sub-sequence, previous val (float)
                loop_vars=[elems - 1, 0, 0, tf.zeros_like(values[-1]), decayed_values],
                back_prop=False
            )

            decayed_values = decayed_values.stack()
            return tf.stop_gradient(tf.reverse(tensor=decayed_values, axis=[0]))
        elif get_backend() == "pytorch":
            # Scan all sequences in reverse:
            discounted = []
            i = 0
            length = 0
            prev_v = 0
            for v in reversed(values):
                # Arrived at new sequence, start over.
                if sequence_indices[i] == 1:
                    length = 0
                    prev_v = 0

                # Accumulate prior value.
                accum_v = prev_v + v * pow(decay, length)
                discounted.append(accum_v)
                prev_v = accum_v

                # Increase length of current sub-sequence.
                length += 1

            # Reverse, convert, and return final.
            return torch.tensor(list(reversed(discounted)), dtype=torch.float32)

    @rlgraph_api(must_be_complete=False)
    def _graph_fn_bootstrap_values(self, values, sequence_indices):
        """
        Inserts value estimates at the end of each sub-sequence for a given sequence. That is,
        0 is inserted after teach terminal and the final value of the sub-sequence if the sequence does not
        end with a terminal.

        Args:
            values (DataOp): Values to adjust.
            sequence_indices (DataOp): Indices denoting sequences, e.g. terminal values.

        Returns:
            Bootstrapped sequence.
        """
        if get_backend() == "tf":
            num_values = tf.shape(values)[0]
            bootstrap_value = values[-1]
            adjusted_values = tf.TensorArray(dtype=tf.float32, infer_shape=False,
                                             size=1, dynamic_size=True, clear_after_read=False)

            def write(write_index, adjusted_values, value):
                adjusted_values = adjusted_values.write(write_index, value)
                write_index += 1
                return adjusted_values, write_index

            def body(index, write_index, adjusted_values):
                adjusted_values = adjusted_values.write(write_index, values[index])
                write_index += 1

                # Append 0 whenever we terminate.
                adjusted_values, write_index = tf.cond(
                    pred=tf.equal(sequence_indices[index], 1),
                    true_fn=lambda: write(write_index, adjusted_values, 0.0),
                    false_fn=lambda: (adjusted_values, write_index)
                )
                return index + 1, write_index, adjusted_values

            def cond(index, write_index, adjusted_values):
                return index < num_values

            index, write_index, adjusted_values = tf.while_loop(
                cond=cond,
                body=body,
                loop_vars=[0, 0, adjusted_values],
                back_prop=False
            )

            # In case the last element was not a terminal, append boot_strap_value.
            # If was terminal -> already appended in loop.
            adjusted_values, _ = tf.cond(pred=tf.greater(sequence_indices[-1], 0),
                                         true_fn=lambda: (adjusted_values, write_index),
                                         false_fn=lambda: write(write_index, adjusted_values, bootstrap_value))

            return adjusted_values.stack()
        elif get_backend() == "pytorch":
            adjusted = []
            for i, val in enumerate(values.tolist()):
                adjusted.append(val)

                # Append 0 as next val.
                if sequence_indices[i] == 1:
                    adjusted.append(0)

            # If terminal, we already appended.
            if sequence_indices[-1] == 0:
                # Append last value.
                adjusted.append(values[-1])

            # Convert and return.
            return torch.tensor(adjusted)

