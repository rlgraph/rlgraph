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

from rlgraph.components.layers.preprocessing.preprocess_layer import PreprocessLayer
from rlgraph.spaces.float_box import FloatBox
from rlgraph.utils.decorators import rlgraph_api
from rlgraph.utils.ops import unflatten_op


class Multiply(PreprocessLayer):
    """
    Scales an input by a constant scaling-factor.
    """
    def __init__(self, factor, scope="multiply", **kwargs):
        """
        Args:
            scaling_factor (float): The factor to scale with.
        """
        super(Multiply, self).__init__(scope=scope, **kwargs)
        self.factor = factor

    def get_preprocessed_space(self, space):
        # Translate to corresponding FloatBoxes.
        ret = dict()
        for key, value in space.flatten().items():
            ret[key] = FloatBox(shape=value.shape, add_batch_rank=value.has_batch_rank,
                                add_time_rank=value.has_time_rank)
        return unflatten_op(ret)

    @rlgraph_api(flatten_ops=True, split_ops=True)
    def _graph_fn_call(self, inputs):
        """
        Multiplies the input with our factor.

        Args:
            inputs (tensor): The input to be scaled.

        Returns:
            op: The op to scale the input.
        """
        result = inputs * self.factor
        # TODO: Move into util function.
        if hasattr(inputs, "_batch_rank"):
            result._batch_rank = inputs._batch_rank
        if hasattr(inputs, "_time_rank"):
            result._time_rank = inputs._time_rank
        return result


class Divide(PreprocessLayer):
    """
    Divides an input by a constant value.
    """
    def __init__(self, divisor, scope="divide", **kwargs):
        """
        Args:
            scaling_factor (float): The factor to scale with.
        """
        super(Divide, self).__init__(scope=scope, **kwargs)
        self.divisor = divisor

    def get_preprocessed_space(self, space):
        # Translate to corresponding FloatBoxes.
        ret = dict()
        for key, value in space.flatten().items():
            ret[key] = FloatBox(shape=value.shape, add_batch_rank=value.has_batch_rank,
                                add_time_rank=value.has_time_rank)
        return unflatten_op(ret)

    @rlgraph_api(flatten_ops=True, split_ops=True)
    def _graph_fn_call(self, inputs):
        """
        Divides the input by with our divisor.

        Args:
            inputs (tensor): The input to be divided.

        Returns:
            DataOp: The op to divide the input.
        """
        result = inputs / self.divisor
        # TODO: Move into util function.
        if hasattr(inputs, "_batch_rank"):
            result._batch_rank = inputs._batch_rank
        if hasattr(inputs, "_time_rank"):
            result._time_rank = inputs._time_rank
        return result

