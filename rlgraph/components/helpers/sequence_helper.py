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
        super(SequenceHelper, self).__init__(space_agnostic=True, scope=scope, **kwargs)

    @rlgraph_api
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
                    pred=tf.equal(sequence_indices[index], tf.ones_like(sequence_indices[index])),
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
    def _graph_fn_calc_sequence_decays(self, sequence_indices, decay=0.9):
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
            tuple:
                - Sequence lengths.
                - Decays.
        """
        if get_backend() == "tf":
            elems = tf.shape(input=sequence_indices)[0]
            sequence_indices = tf.cast(sequence_indices, dtype=tf.int32)

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
            return torch.tensor(sequence_lengths, dtype=torch.int32), torch.tensor(decays, dtype=torch.int32)

    @rlgraph_api
    def _graph_fn_reverse_apply_decays_to_sequence(self, values, sequence_indices, decay=0.9):
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
            sequence_indices = tf.cast(sequence_indices, dtype=tf.int32)

            def insert_body(index, forward_index, write_index, length, prev_v, decayed_values):
                # Decay is based on length, so val = decay^length
                decay_val = tf.pow(x=decay, y=tf.cast(length, dtype=tf.float32))
                accum_v = prev_v + values[index] * decay_val

                # Write decayed val into array.
                decayed_values = decayed_values.write(write_index, accum_v)
                prev_v = accum_v

                # Reset length to 0 if terminal encountered.
                # NOTE: We cannot prev_v to 0.0 because values[index] might have a more complex shape,
                # so this violates shape checks.
                length, prev_v = tf.cond(
                    pred=tf.equal(sequence_indices[forward_index], tf.ones_like(sequence_indices[forward_index])),
                    true_fn=lambda: (0, tf.zeros_like(prev_v)),
                    false_fn=lambda: (length, prev_v)
                )
                # index = tf.Print(index, [index, prev_v], summarize=100, message="index, prev = ")

                # Increase write-index and length of sub-sequence, decrease loop index in reverse iteration.
                return index - 1, forward_index + 1, write_index + 1, length + 1, prev_v, decayed_values

            def cond(index, forward_index, write_index, length, prev_v, decayed_values):
                # Scan in reverse.
                return index >= 0

            _, _, _, _, _, decayed_values = tf.while_loop(
                cond=cond,
                body=insert_body,
                # loop index, index writing to tensor array, current length of sub-sequence, previous val (float)
                loop_vars=[elems - 1, 0, 0, 0, tf.zeros_like(values[-1]), decayed_values],
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
            for v in reversed(values.data):
                # Accumulate prior value.
                accum_v = prev_v + v * pow(decay, length)
                discounted.append(accum_v)
                prev_v = accum_v

                # Arrived at new sequence, start over.
                if sequence_indices[i] == 1:
                    length = 0
                    prev_v = 0

                # Increase length of current sub-sequence.
                length += 1
                i += 1

            # Reverse, convert, and return final.
            return torch.tensor(list(reversed(discounted)), dtype=torch.float32)

    @rlgraph_api
    def _graph_fn_bootstrap_values(self, rewards, values, terminals, sequence_indices, discount=0.99):
        """
        Inserts value estimates at the end of each sub-sequence for a given sequence and computes deltas
        for generalized advantage estimation. That is, 0 is inserted after teach terminal and the final value of the
        sub-sequence if the sequence does not end with a terminal. We then compute for each subsequence

        delta = reward + discount * bootstrapped_values[1:] - bootstrapped_values[:-1]


        Args:
            rewards (DataOp): Rewards for the observed sequences.
            values (DataOp): Value estimates for the observed sequences.
            terminals (DataOp): Terminals in sequences
            sequence_indices (DataOp): Int indices denoting sequences (which may be non-terminal episode fragments
                from multiple environments.
            discount (float): Discount to apply to delta computation.

        Returns:
            Sequence of deltas.
        """
        if get_backend() == "tf":
            values = tf.squeeze(input=values)
            num_values = tf.shape(input=values)[0]

            # Again ensure last index is 1 for any sub-sample arriving here.
            last_sequence = tf.expand_dims(sequence_indices[-1], -1)
            sequence_indices = tf.concat([sequence_indices[:-1], tf.ones_like(last_sequence, dtype=tf.bool)], axis=0)

            # Cannot use 0.0 because unknown shape.
            deltas = tf.TensorArray(dtype=tf.float32, infer_shape=False, size=num_values,
                                    dynamic_size=False, clear_after_read=False, name="bootstrap-deltas")

            # Boot-strap with 0 only if terminals[i] and sequence_indices[i] are both true.
            boot_strap_zeros = tf.where(
                condition=tf.logical_and(sequence_indices, terminals),
                x=tf.ones_like(sequence_indices),
                y=tf.zeros_like(sequence_indices),
            )

            def write(index, deltas, start_index):
                # First: Concat the slice of values representing the current sequence with bootstrap value.
                baseline_slice = values[start_index:index + 1]
                # Expand so value has a batch dim when we concat.

                # If true terminal, append 0. Otherwise, append boot-strap val -> last observed val.
                bootstrap_value = tf.cond(
                    pred=tf.equal(boot_strap_zeros[index], tf.ones_like(sequence_indices[index])),
                    true_fn=lambda: tf.zeros_like(tensor=values[index], dtype=tf.float32),
                    false_fn=lambda: values[index],
                )

                value = tf.expand_dims(bootstrap_value, 0)
                adjusted_v = tf.concat([baseline_slice, value], axis=0)

                # Compute deltas for this sequence.
                sequence_deltas = rewards[start_index:index + 1] + discount * adjusted_v[1:] - adjusted_v[:-1]

                # Write delta to tensor-array.
                write_indices = tf.range(start=start_index, limit=index + 1)
                deltas = deltas.scatter(write_indices, sequence_deltas)

                start_index = index + 1
                # Set start-index for the next sub-sequence to index + 1
                return deltas, start_index

            def body(index, start_index, deltas):
                # Whenever we encounter a sequence end, we compute deltas for that sequence.
                deltas, start_index = tf.cond(
                    pred=tf.equal(sequence_indices[index], tf.ones_like(sequence_indices[index])),
                    true_fn=lambda: write(index, deltas, start_index),
                    false_fn=lambda: (deltas, start_index)
                )
                return index + 1, start_index, deltas

            def cond(index, start_index, deltas):
                return index < num_values

            index, start_index, deltas = tf.while_loop(
                cond=cond,
                body=body,
                loop_vars=[0, 0, deltas],
                parallel_iterations=1,
                back_prop=False
            )

            deltas = deltas.stack()
            # Squeeze because we inserted
            return tf.squeeze(deltas)
        elif get_backend() == "pytorch":
            deltas = []
            discount_tensor = torch.tensor(discount)
            start_index = 0
            i = 0
            if len(values) > 1:
                last_sequence = torch.unsqueeze(sequence_indices[-1], -1)
                sequence_indices = torch.cat((sequence_indices[:-1], torch.ones_like(last_sequence)), 0)

            for _ in range(len(values)):
                if sequence_indices[i]:
                    # Compute deltas for this sub-sequence.
                    # Cannot do this all at once because we would need the correct offsets for each sub-sequence.
                    baseline_slice = list(values[start_index:i + 1])
                    if terminals[i]:
                        baseline_slice.append(0)
                    else:
                        baseline_slice.append(values[-1])
                    adjusted_v = torch.tensor(baseline_slice)

                    # +1 because we want to include i-th value.
                    delta = rewards[start_index:i + 1] + discount_tensor * adjusted_v[1:] - adjusted_v[:-1]
                    deltas.extend(delta)
                    start_index = i + 1
                i += 1

            return torch.tensor(deltas)

