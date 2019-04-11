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
from rlgraph.components.layers.preprocessing.preprocess_layer import PreprocessLayer
from rlgraph.utils.decorators import rlgraph_api
from rlgraph.utils.ops import unflatten_op

if get_backend() == "tf":
    import tensorflow as tf
elif get_backend() == "pytorch":
    import torch


class Transpose(PreprocessLayer):
    """
    """
    def __init__(self, output_is_time_major=True, scope="transpose", **kwargs):
        """
        Args:
            output_is_time_major (Optional[bool]): Whether the output of this Component will always be time-major.
                If None, get this information from input-spaces. If given, this Component will be space-agnostic.
                Default: True (batch-major -> time-major transpose).
        """
        super(Transpose, self).__init__(space_agnostic=(output_is_time_major is not None), scope=scope, **kwargs)

        # Overrides everything in dict: `self.output_time_majors`.
        self.output_is_time_major = output_is_time_major

        # Only used if `self.output_is_time_major` is None.
        self.output_time_majors = dict()

    def create_variables(self, input_spaces, action_space=None):
        in_space = input_spaces["inputs"]  # type: Space
        # Make sure output time_majors are stored.
        self.get_preprocessed_space(in_space)

    def get_preprocessed_space(self, space):
        ret = dict()
        for key, single_space in space.flatten().items():
            class_ = type(single_space)
            # We flip batch and time ranks.
            time_major = not single_space.time_major
            ret[key] = class_(shape=single_space.shape,
                              add_batch_rank=single_space.has_batch_rank,
                              add_time_rank=single_space.has_time_rank, time_major=time_major)
            self.output_time_majors[key] = time_major
        ret = unflatten_op(ret)
        return ret

    @rlgraph_api(flatten_ops=True, split_ops=True, add_auto_key_as_first_param=True)
    def _graph_fn_call(self, key, inputs):
        """
        Transposes the input by flipping batch and time ranks.
        """
        if get_backend() == "tf":
            # Flip around ranks 0 and 1.
            transposed = tf.transpose(
                inputs,
                perm=(1, 0) + tuple(i for i in range(2, len(inputs.shape.as_list()))), name="transpose"
            )
            if self.output_is_time_major is None:
                transposed._time_rank = 0 if self.output_time_majors[key] is True else 1
                transposed._batch_rank = 0 if self.output_time_majors[key] is False else 1
            else:
                transposed._time_rank = 0 if self.output_is_time_major is True else 1
                transposed._batch_rank = 0 if self.output_is_time_major is False else 1

            return transposed

        elif get_backend() == "pytorch":
            perm = (1, 0) + tuple(i for i in range(2, len(list(inputs.shape))))
            return torch.transpose(inputs, perm)
