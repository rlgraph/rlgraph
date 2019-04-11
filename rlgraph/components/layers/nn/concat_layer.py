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
from rlgraph.components.layers.nn.nn_layer import NNLayer
from rlgraph.utils.decorators import rlgraph_api
from rlgraph.utils.util import force_list

if get_backend() == "tf":
    import tensorflow as tf
elif get_backend() == "pytorch":
    import torch


class ConcatLayer(NNLayer):
    """
    A simple concatenation layer wrapper. The ConcatLayer is a Layer without sub-components but with n
    api_methods and 1 output, where input data is concatenated into one output by its GraphFunction.
    """
    def __init__(self, axis=-1, dict_keys=None, scope="concat-layer", **kwargs):
        """
        Args:
            axis (int): The axis along which to concatenate. Use negative numbers to count from end.
                All api_methods to this layer must have the same shape, except for the `axis` rank.
                Default: -1.

            dict_keys (Optional[List[str]]): An optional list of dict keys to use to retrieve input data from an
                incoming dict (instead of a tuple series of inputs).
        """
        super(ConcatLayer, self).__init__(scope=scope, **kwargs)

        self.axis = axis
        self.dict_keys = dict_keys

        # Whether input spaces are time-major or not.
        self.time_major = None

    def check_input_spaces(self, input_spaces, action_space=None):
        super(ConcatLayer, self).check_input_spaces(input_spaces, action_space)
        # Make sure all inputs have the same shape except for the last rank.
        if self.dict_keys:
            self.in_space_0 = input_spaces["inputs[0]"][self.dict_keys[0]]
        else:
            self.in_space_0 = input_spaces["inputs[0]"]
        self.time_major = self.in_space_0.time_major
        # Loop through either all args or all kwargs Spaces.
        idx = 0
        while self.dict_keys is None or len(self.dict_keys) > idx:
            if self.dict_keys is None:
                key = "inputs[{}]".format(idx)
                if key not in input_spaces:
                    break
                in_space = input_spaces[key]
            else:
                in_space = input_spaces["inputs[0]"][self.dict_keys[idx]]

            # Make sure the shapes match (except for last rank).
            assert self.in_space_0.shape[:-1] == in_space.shape[:-1], \
                "ERROR: Input spaces to ConcatLayer must have same shape except for last rank! " \
                "0th input's shape is {}, but {}st input's shape is {} (all shapes here are without " \
                "batch/time-ranks).".format(self.in_space_0.shape, idx, in_space.shape)
            idx += 1

    @rlgraph_api
    def _graph_fn_call(self, *inputs):
        # Simple translation from dict to tuple-input.
        if self.dict_keys is not None:
            inputs = [inputs[0][key] for key in self.dict_keys]

        if get_backend() == "tf":
            concat_output = tf.concat(values=inputs, axis=self.axis)
            # Add batch/time-rank information.
            concat_output._batch_rank = 0 if self.time_major is False else 1
            if self.in_space_0.has_time_rank:
                concat_output._time_rank = 0 if self.time_major is True else 1
            return concat_output
        elif get_backend() == "pytorch":
            return torch.cat(force_list(inputs))
