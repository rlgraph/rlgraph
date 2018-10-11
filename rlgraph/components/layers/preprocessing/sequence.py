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

from collections import deque

import numpy as np
from six.moves import xrange as range_

from rlgraph import get_backend
from rlgraph.spaces.space_utils import sanity_check_space
from rlgraph.utils.decorators import rlgraph_api
from rlgraph.utils.ops import FlattenedDataOp, unflatten_op
from rlgraph.utils.util import get_rank, get_shape, force_list
from rlgraph.components.layers.preprocessing import PreprocessLayer

if get_backend() == "tf":
    import tensorflow as tf
elif get_backend() == "pytorch":
    import torch


class Sequence(PreprocessLayer):
    """
    Concatenate `length` state vectors. Example: Used in Atari
    problems to create the Markov property (velocity of game objects as they move across the screen).
    """

    def __init__(self, sequence_length=2, batch_size=1, add_rank=True, in_data_format="channels_last",
                 out_data_format="channels_last", scope="sequence",  **kwargs):
        """
        Args:
            sequence_length (int): The number of records to always concatenate together within the last rank or
                in an extra (added) rank.
            batch_size (int): The batch size for incoming records so multiple inputs can be passed through at once.
            in_data_format (str): One of 'channels_last' (default) or 'channels_first'. Specifies which rank (first or
                last) is the color-channel. If the input Space is with batch, the batch always has the first rank.
            out_data_format (str): One of 'channels_last' (default) or 'channels_first'. Specifies which rank (first or
                last) is the color-channel in output. If the input Space is with batch,
                 the batch always has the first rank.
            add_rank (bool): Whether to add another rank to the end of the input with dim=length-of-the-sequence.
                If False, concatenates the sequence within the last rank.
                Default: True.
        """
        # Switch off split (it's switched on for all LayerComponents by default).
        # -> accept any Space -> flatten to OrderedDict -> input & return OrderedDict -> re-nest.
        super(Sequence, self).__init__(scope=scope, **kwargs)

        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.add_rank = add_rank

        self.in_data_format = in_data_format
        if get_backend() == "pytorch":
            # Always channels first for PyTorch.
            self.out_data_format = "channels_first"
        else:
            self.out_data_format = out_data_format

        # The sequence-buffer where we store previous inputs.
        self.buffer = None
        # The index into the buffer's.
        self.index = None
        # The output spaces after preprocessing (per flat-key).
        self.output_spaces = None
        if self.backend == "python" or get_backend() == "python" or get_backend() == "pytorch":
            self.deque = deque([], maxlen=self.sequence_length)

    def get_preprocessed_space(self, space):
        ret = dict()
        for key, value in space.flatten().items():
            shape = list(value.shape)
            if self.add_rank:
                shape.append(self.sequence_length)
            else:
                shape[-1] *= self.sequence_length

            # TODO move to transpose component.
            # Transpose.
            if self.in_data_format == "channels_last" and self.out_data_format == "channels_first":
                shape.reverse()
                ret[key] = value.__class__(shape=tuple(shape), add_batch_rank=value.has_batch_rank)
            else:
                ret[key] = value.__class__(shape=tuple(shape), add_batch_rank=value.has_batch_rank)
        return unflatten_op(ret)

    def check_input_spaces(self, input_spaces, action_space=None):
        super(Sequence, self).check_input_spaces(input_spaces, action_space)
        in_space = input_spaces["preprocessing_inputs"]

        # Require preprocessing_inputs to not have time rank (batch rank doesn't matter).
        sanity_check_space(in_space, must_have_time_rank=False)

    def create_variables(self, input_spaces, action_space=None):
        in_space = input_spaces["preprocessing_inputs"]
        self.output_spaces = self.get_preprocessed_space(in_space)

        self.index = self.get_variable(name="index", dtype="int", initializer=-1, trainable=False)
        if self.backend == "python" or get_backend() == "python" or get_backend() == "pytorch":
            # TODO get variable does not return an int for python
            self.index = -1
        else:
            self.buffer = self.get_variable(
                name="buffer", trainable=False, from_space=in_space,
                add_batch_rank=self.batch_size if in_space.has_batch_rank is not False else False,
                add_time_rank=self.sequence_length, time_major=True, flatten=True
            )

    @rlgraph_api
    def _graph_fn_reset(self):
        if self.backend == "python" or get_backend() == "python" or get_backend() == "pytorch":
            self.index = -1
        elif get_backend() == "tf":
            return tf.variables_initializer([self.index])

    @rlgraph_api(flatten_ops=True, split_ops=False)
    def _graph_fn_apply(self, preprocessing_inputs):
        """
        Sequences (stitches) together the incoming inputs by using our buffer (with stored older records).
        Sequencing happens within the last rank if `self.add_rank` is False, otherwise a new rank is added at the end
        for the sequencing.

        Args:
            preprocessing_inputs (FlattenedDataOp): The FlattenedDataOp to be sequenced.
                One sequence is generated separately for each SingleDataOp in api_methods.

        Returns:
            FlattenedDataOp: The FlattenedDataOp holding the sequenced SingleDataOps as values.
        """
        # A normal (index != -1) assign op.
        if self.backend == "python" or get_backend() == "python":
            if self.index == -1:
                for _ in range_(self.sequence_length):
                    self.deque.append(preprocessing_inputs)
            else:
                self.deque.append(preprocessing_inputs)
            self.index = (self.index + 1) % self.sequence_length

            if self.add_rank:
                sequence = np.stack(self.deque, axis=-1)
            # Concat the sequence items in the last rank.
            else:
                sequence = np.concatenate(self.deque, axis=-1)

            # TODO move into transpose component.
            if self.in_data_format == "channels_last" and self.out_data_format == "channels_first":
                sequence = sequence.transpose((0, 3, 2, 1))

            return sequence
        elif get_backend() == "pytorch":
            if self.index == -1:
                for _ in range_(self.sequence_length):
                    if isinstance(preprocessing_inputs, dict):
                        for key, value in preprocessing_inputs.items():
                            self.deque.append(value)
                    else:
                        self.deque.append(preprocessing_inputs)
            else:
                if isinstance(preprocessing_inputs, dict):
                    for key, value in preprocessing_inputs.items():
                        self.deque.append(value)
                        self.index = (self.index + 1) % self.sequence_length
                else:
                    self.deque.append(preprocessing_inputs)
                    self.index = (self.index + 1) % self.sequence_length

            if self.add_rank:
                sequence = torch.stack(torch.tensor(self.deque), dim=-1)
            # Concat the sequence items in the last rank.
            else:
                sequence = torch.cat([torch.tensor(t) for t in self.deque], dim=-1)

            # TODO remove when transpose component implemented.
            if self.in_data_format == "channels_last" and self.out_data_format == "channels_first":
                # Problem: PyTorch does not have data format options in conv layers ->
                # only channels first supported.
                # -> Confusingly have to transpose.
                # B W H C -> B C W H
                # e.g. atari: [4 84 84 4] -> [4 4 84 84]
                sequence = sequence.permute(0, 3, 2, 1)

            return sequence
        elif get_backend() == "tf":
            # Assigns the input_ into the buffer at the current time index.
            def normal_assign():
                assigns = list()
                for key_, value in preprocessing_inputs.items():
                    assign_op = self.assign_variable(ref=self.buffer[key_][self.index], value=value)
                    assigns.append(assign_op)
                return assigns

            # After a reset (time index is -1), fill the entire buffer with `self.sequence_length` x input_.
            def after_reset_assign():
                assigns = list()
                for key_, value in preprocessing_inputs.items():
                    multiples = (self.sequence_length,) + tuple([1] * get_rank(value))
                    input_ = tf.expand_dims(input=value, axis=0)
                    assign_op = self.assign_variable(
                        ref=self.buffer[key_], value=tf.tile(input=input_, multiples=multiples)
                    )
                    assigns.append(assign_op)
                return assigns

            # Insert the input at the correct index or fill empty buffer entirely with input.
            insert_inputs = tf.cond(pred=(self.index >= 0), true_fn=normal_assign, false_fn=after_reset_assign)

            # Make sure the input has been inserted.
            with tf.control_dependencies(control_inputs=force_list(insert_inputs)):
                # Then increase index by 1.
                index_plus_1 = self.assign_variable(ref=self.index, value=((self.index + 1) % self.sequence_length))

            # Then gather the output.
            with tf.control_dependencies(control_inputs=[index_plus_1]):
                sequences = FlattenedDataOp()
                # Collect the correct previous inputs from the buffer to form the output sequence.
                for key in preprocessing_inputs.keys():
                    n_in = [self.buffer[key][(self.index + n) % self.sequence_length]
                            for n in range_(self.sequence_length)]

                    # Add the sequence-rank to the end of our inputs.
                    if self.add_rank:
                        sequence = tf.stack(values=n_in, axis=-1)
                    # Concat the sequence items in the last rank.
                    else:
                        sequence = tf.concat(values=n_in, axis=-1)

                    # Must pass the sequence through a placeholder_with_default dummy to set back the
                    # batch rank to '?', instead of 1 (1 would confuse the auto Space inference).
                    sequences[key] = tf.placeholder_with_default(
                        sequence, shape=(None,) + tuple(get_shape(sequence)[1:])
                    )
            # TODO implement transpose
                return sequences

