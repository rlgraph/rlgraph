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

from six.moves import xrange as range_

import tensorflow as tf

from rlgraph.utils.ops import FlattenedDataOp, unflatten_op
from rlgraph.utils.util import get_rank, get_shape, force_list, get_batch_size
from rlgraph.components.layers.preprocessing import PreprocessLayer


class Sequence(PreprocessLayer):
    """
    Concatenate `length` state vectors. Example: Used in Atari
    problems to create the Markov property (velocity of game objects as they move across the screen).
    """

    def __init__(self, length=2, add_rank=True, scope="sequence", **kwargs):
        """
        Args:
            length (int): The number of records to always concatenate together.
                The very first record is simply repeated `length` times.
                The second record will generate: Itself and `length`-1 times the very first record.
                Etc..
            add_rank (bool): Whether to add another rank to the end of the input with dim=length-of-the-sequence.
                If False, concatenates the sequence within the last rank.
                Default: True.
        """
        # Switch off split (it's switched on for all LayerComponents by default).
        # -> accept any Space -> flatten to OrderedDict -> input & return OrderedDict -> re-nest.
        super(Sequence, self).__init__(scope=scope, split_ops=False, **kwargs)

        self.length = length
        self.add_rank = add_rank

        # Whether the first rank of the api_methods is the batch dimension (known at build time).
        self.first_rank_is_batch = None
        # The sequence-buffer where we store previous api_methods.
        self.buffer = None
        # The index into the buffer.
        self.index = None
        # The output spaces after preprocessing (per flat-key).
        self.output_spaces = None

    def get_preprocessed_space(self, space):
        ret = dict()
        for key, value in space.flatten().items():
            shape = list(value.shape)
            if self.add_rank:
                shape.append(self.length)
            else:
                shape[-1] *= self.length
            ret[key] = value.__class__(shape=tuple(shape), add_batch_rank=value.has_batch_rank)
        return unflatten_op(ret)

    def check_input_spaces(self, input_spaces, action_space):
        super(Sequence, self).check_input_spaces(input_spaces, action_space)
        in_space = input_spaces["apply"][0]
        self.first_rank_is_batch = in_space.has_batch_rank

        self.output_spaces = self.get_preprocessed_space(in_space)

    def create_variables(self, input_spaces, action_space):
        in_space = input_spaces["apply"][0]
        self.first_rank_is_batch = in_space.has_batch_rank

        # Cut the "batch rank" (always 1 anyway) and replace it with the "sequence-rank".
        self.buffer = self.get_variable(name="buffer", trainable=False,
                                        from_space=in_space, add_batch_rank=self.length,
                                        flatten=True)
        # Our index. Points to the slot where we insert next (-1 after reset).
        self.index = self.get_variable(name="index", dtype="int", initializer=-1, trainable=False)

    def _graph_fn_reset(self):
        return tf.variables_initializer([self.index])

    def _graph_fn_apply(self, inputs):
        """
        Sequences (stitches) together the incoming inputs by using our buffer (with stored older records).
        Sequencing happens within the last rank if `self.add_rank` is False, otherwise a new rank is added at the end
        for the sequencing.

        Args:
            inputs (FlattenedDataOp): The FlattenedDataOp to be sequenced.
                One sequence is generated separately for each SingleDataOp in api_methods.

        Returns:
            FlattenedDataOp: The FlattenedDataOp holding the sequenced SingleDataOps as values.
        """
        # A normal (index != -1) assign op.
        def normal_assign():
            assigns = list()
            for key, value in inputs.items():
                # [0]=skip batch (which must be len=1 anyway)
                assigns.append(self.assign_variable(
                    ref=self.buffer[key][self.index], value=value[0] if self.first_rank_is_batch else value)
                )
            return assigns

        # If index is still -1 (after reset):
        # Pre-fill the entire buffer with `self.length` x input_.
        def after_reset_assign():
            assigns = list()
            for key, value in inputs.items():
                multiples = (self.length,) + tuple([1] * (get_rank(value) - (1 if self.first_rank_is_batch else 0)))
                in_ = value if self.first_rank_is_batch else tf.expand_dims(value, 0)
                assigns.append(self.assign_variable(
                    ref=self.buffer[key],
                    value=tf.tile(input=in_, multiples=multiples)
                ))
            return assigns

        # Insert the input at the correct index or fill empty buffer entirely with input.
        insert_inputs = tf.cond(pred=(self.index >= 0), true_fn=normal_assign, false_fn=after_reset_assign)

        # Make sure the input has been inserted.
        with tf.control_dependencies(control_inputs=force_list(insert_inputs)):
            # Then increase index by 1.
            index_plus_1 = self.assign_variable(ref=self.index, value=((self.index + 1) % self.length))

        # Then gather the output.
        with tf.control_dependencies(control_inputs=[index_plus_1]):
            sequences = FlattenedDataOp()
            # Collect the correct previous inputs from the buffer to form the output sequence.
            for key in inputs.keys():
                n_in = [self.buffer[key][(self.index + n) % self.length] for n in range_(self.length)]

                # Add the sequence-rank to the end of our api_methods.
                if self.add_rank:
                    sequence = tf.stack(values=n_in, axis=-1)
                # Concat the sequence items in the last rank.
                else:
                    sequence = tf.concat(values=n_in, axis=-1)

                # Put batch rank back in (buffer does not have it).
                if self.first_rank_is_batch:
                    sequence = tf.expand_dims(input=sequence, axis=0, name="apply")
                # Or not.
                else:
                    sequence = tf.identity(input=sequence, name="apply")

                # Must pass the sequence through a placeholder_with_default dummy to set back the
                # batch rank to '?', instead of 1 (1 would confuse the auto Space inference).
                sequences[key] = tf.placeholder_with_default(sequence, shape=(None,) + tuple(get_shape(sequence)[1:]))

            return sequences
