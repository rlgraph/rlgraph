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

from collections import OrderedDict

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
        # Switch off split again (it's switched on for all layers by default).
        # -> accept OrderedDict, return OrderedDict (which will then be re-nested).
        super(Sequence, self).__init__(scope=scope, computation_settings=dict(split_container_spaces=False),
                                       **kwargs)

        self.sequence_length = seq_length
        self.add_rank = add_rank

        # Whether the first rank of the inputs is the batch dimension (known at create_variables time).
        self.first_rank_is_batch = None
        # The sequence-buffer where we store previous inputs.
        self.buffer = None
        ## The last flattened key (signal to bump index by 1).
        #self.last_key = None
        # The index into the buffer.
        self.index = None
        # Op to bump the index by 1 (round-robin).
        self.index_plus_1 = None

    def create_variables(self, input_spaces):
        # Cut the "batch rank" (always 1 anyway) and replace it with the "sequence-rank".
        in_space = input_spaces["input"]
        self.first_rank_is_batch = in_space.has_batch_rank
        self.buffer = self.get_variable(name="buffer", trainable=False,
                                        from_space=in_space, add_batch_rank=self.sequence_length,
                                        flatten=True)
        #self.last_key = next(reversed(self.buffer))
        self.index = self.get_variable(name="index", dtype="int", initializer=-1, trainable=False)

    def _computation_reset(self):
        return tf.variables_initializer([self.index])

    def _computation_apply(self, inputs):
        """
        Stitches together the incoming inputs by using our buffer (with stored older records).

        Args:
            #flat_key (str): The key of the current input op in the flattened Space.
            inputs (OrderedDict): The flattened dict of input ops to sequence (one sequence is generated separately
                for each item in inputs_).
        """
        # A normal (index != -1) assign op.
        def normal_assign():
            assigns = list()
            for key, value in inputs:
                # [0]=skip batch (which must be len=1 anyway)
                assigns.append(tf.assign(ref=self.buffer[key][self.index], value=value[0]))
            return tf.group(*assigns)

        # If index is still -1 (after reset):
        # Pre-fill the entire buffer with `self.sequence_length` x input_.
        def after_reset_assign():
            assigns = list()
            for key, value in inputs:
                multiples = (self.sequence_length,) + tuple([1] * (get_rank(inputs[key]) - 1))
                assigns.append(tf.assign(ref=self.buffer[key], value=tf.tile(input=inputs[key], multiples=multiples)))
            return tf.group(*assigns)

        # Insert the input at the correct index or fill empty buffer entirely with input.
        insert_inputs = tf.cond(pred=(self.index >= 0), true_fn=normal_assign, false_fn=after_reset_assign)

        # Make sure the input has been inserted ..
        # .. and that the first rank's dynamic size is 1 (single item, no batching).
        dependencies = [insert_inputs]
        if self.first_rank_is_batch:
            for key, value in inputs:
                dependencies.append(([tf.assert_equal(x=tf.shape(input=value)[0], y=1)]))
        with tf.control_dependencies(control_inputs=dependencies):
            n_inputs = OrderedDict()
            # Collect the correct previous inputs from the buffer to form the output sequence.
            for key in inputs.keys():
                n_in = [self.buffer[key][(self.index - n - 1) % self.sequence_length]
                        for n in range(self.sequence_length)]

                # Add the sequence-rank to the end of our inputs.
                if self.add_rank:
                    n_inputs[key] = tf.stack(values=n_in, axis=-1)
                # Concat the sequence items in the last rank.
                else:
                    n_inputs[key] = tf.concat(values=n_in, axis=-1)

        # Increase index by 1.
        index_plus_1 = tf.assign(ref=self.index,
                                 value=((tf.maximum(x=self.index, y=0) + 1) % self.sequence_length))

        with tf.control_dependencies(control_inputs=[index_plus_1]):
            out = OrderedDict()
            for key, value in n_inputs:
                # Put batch rank back in (buffer does not have it).
                if self.first_rank_is_batch:
                    out[key] = tf.expand_dims(input=value, axis=0, name="apply")
                # Or not.
                else:
                    out[key] = tf.identity(input=value, name="apply")

            return out
