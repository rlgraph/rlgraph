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
from rlgraph.utils.util import force_list

from rlgraph.components.layers.preprocessing.preprocess_layer import PreprocessLayer

if get_backend() == "tf":
    import tensorflow as tf
elif get_backend() == "pytorch":
    import torch.nn as nn


class Concat(PreprocessLayer):
    """
    A simple concatenation layer wrapper. The ConcatLayer is a Layer without sub-components but with n
    api_methods and 1 output, where the in-Sockets's data are concatenated into one out-Socket by its GraphFunction.
    """
    def __init__(self, axis=-1, scope="concat", **kwargs):
        """
        Args:
            axis (int): The axis along which to concatenate. Use negative numbers to count from end.
                All api_methods to this layer must have the same shape, except for the `axis` rank.
                Default: -1.
        """
        super(Concat, self).__init__(flatten_ops=True, split_ops=True,
                                     scope=scope, **kwargs)
        self.axis = axis

        # Whether input spaces are time-major or not.
        self.time_major = None

        # Wrapper for backend.
        if get_backend() == "tf":
            self.layer = tf.keras.layers.Concatenate(axis=self.axis)

    def check_input_spaces(self, input_spaces, action_space=None):
        #super(ConcatLayer, self).check_input_spaces(input_spaces, action_space)
        # Make sure all inputs have the same shape except for the last rank.
        self.in_space_0 = input_spaces["inputs[0]"]
        self.time_major = self.in_space_0.time_major

        # TODO: Reinstante these checks from original ConcatLayer (in nn folder) (for container spaces).
        #idx = 0
        #while True:
        #    key = "inputs[{}]".format(idx)
        #    if key not in input_spaces:
        #        break
        #    # Make sure the shapes match (except for last rank).
        #    assert self.in_space_0.shape[:-1] == input_spaces[key].shape[:-1], \
        #        "ERROR: Input spaces to ConcatLayer must have same shape except for last rank! " \
        #        "0th input's shape is {}, but {}st input's shape is {} (all shapes here are without " \
        #        "batch/time-ranks).".format(self.in_space_0.shape, idx, input_spaces[key].shape)
        #    idx += 1

    def _graph_fn_apply(self, *inputs):
        if get_backend() == "tf":
            concat_output = self.layer.apply(force_list(inputs))
            # Add batch/time-rank information.
            concat_output._batch_rank = 0 if self.time_major is False else 1
            if self.in_space_0.has_time_rank:
                concat_output._time_rank = 0 if self.time_major is True else 1
            return concat_output
        elif get_backend() == "pytorch":
            return nn.Sequential(force_list(inputs))
