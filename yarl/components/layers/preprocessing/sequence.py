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

import tensorflow as tf

from yarl.utils.util import get_rank
from .preprocess_layer import PreprocessLayer


class Sequence(PreprocessLayer):
    """
    Concatenate `length` state vectors. Example: Used in Atari
    problems to create the Markov property (velocity of game objects as they move across the screen).
    """

    def __init__(self, seq_length=2, add_rank=True, scope="sequence", **kwargs):
        """
        Args:
            seq_length (int): The number of records to always concatenate together.
                The very first record is simply repeated `sequence_length` times.
                The second record will generate: Itself and `sequence_length`-1 times the very first record.
                Etc..
            add_rank (bool): Whether to add another rank to the end of the input with dim=length-of-the-sequence.
                This could be useful if e.g. a grayscale image of w x h pixels is coming from the env
                (no color channel). The output of the preprocessor would then be of shape [batch] x w x h x [length].
        """
        # As we need a buffer variable, we do not split our computations.
        super(Sequence, self).__init__(scope=scope, split_container_spaces=False, **kwargs)

        self.sequence_length = seq_length
        self.add_rank = add_rank

        # The sequence-buffer where we store previous inputs.
        self.buffer = None
        # The index into the buffer.
        self.index = None
        # Whether the first rank of the inputs is the batch dimension (known at create_variables time).
        self.first_rank_is_batch = None

    def create_variables(self, input_spaces):
        # Cut the "batch rank" (always 1 anyway) and replace it with the "sequence-rank".
        in_space = input_spaces["input"]
        self.first_rank_is_batch = in_space.add_batch_rank
        self.buffer = self.get_variable(name="buffer", trainable=False,
                                        from_space=in_space, add_batch_rank=self.sequence_length)
        self.index = self.get_variable(name="index", dtype="int", initializer=-1, trainable=False)

    def _computation_reset(self):
        return tf.variables_initializer([self.index])

    def _computation_apply(self, inputs):
        """
        Stitches together the incoming (flattened) inputs by using our buffer (with stored older records).

        Args:
            inputs (Union[OrderedDict,op]): The (un-split) flattened input ops.
        """
        # A normal (index != -1) assign op.
        def normal_assign():
            return tf.assign(ref=self.buffer[(self.index + 0)], value=inputs[0])

        # If index is still -1 (after reset):
        # Pre-fill the entire buffer with `self.sequence_length` x input_.
        def after_reset_assign():
            multiples = (self.sequence_length,) + tuple([1] * (get_rank(inputs)-1))
            return tf.assign(ref=self.buffer, value=tf.tile(input=inputs, multiples=multiples))

        # Insert the input at the correct index or fill empty buffer entirely with input.
        insert_input = tf.cond(pred=(self.index >= 0), true_fn=normal_assign, false_fn=after_reset_assign)

        # Make sure the input has been inserted ..
        # .. and that the first rank's dynamic size is 1 (single item, no batching).
        dependencies = [insert_input] + ([tf.assert_equal(x=tf.shape(input=inputs)[0], y=1)] if
                                         self.first_rank_is_batch else [])
        with tf.control_dependencies(control_inputs=tuple(dependencies)):
            # Increase index by 1.
            increment_index = tf.assign(ref=self.index,
                                        value=((tf.maximum(x=self.index, y=0) + 1) % self.sequence_length))

        with tf.control_dependencies(control_inputs=(increment_index,)):
            # Collect the correct previous inputs from the buffer to form the output sequence.
            n_inputs = [self.buffer[(self.index - n - 2) % self.sequence_length]
                                 for n in range(self.sequence_length)]

            # Add the sequence-rank to the end of our inputs.
            if self.add_rank:
                n_inputs = tf.stack(values=n_inputs, axis=-1)
            # Concat the sequence items in the last rank.
            else:
                n_inputs = tf.concat(values=n_inputs, axis=-1)

            # Put batch rank back in (buffer does not have it).
            return tf.expand_dims(input=n_inputs, axis=0)

