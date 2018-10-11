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
from rlgraph.spaces import Space
from rlgraph.components.layers.preprocessing import PreprocessLayer
from rlgraph.utils.decorators import rlgraph_api
from rlgraph.utils.util import SMALL_NUMBER


class Normalize(PreprocessLayer):
    """
    Normalizes an input over all axes individually (denoted as `Xi` below) according to the following formula:

    Xi = (Xi - min(Xi)) / (max(Xi) - min(Xi) + epsilon),
        where:
        Xi is one entire axis of values.
        max(Xi) is the max value along this axis.
        min(Xi) is the min value along this axis.
        epsilon is a very small constant number (to avoid dividing by 0).
    """
    def __init__(self, scope="normalize", **kwargs):
        super(Normalize, self).__init__(scope=scope, **kwargs)
        self.axes = None

    def check_input_spaces(self, input_spaces, action_space=None):
        super(Normalize, self).check_input_spaces(input_spaces, action_space)

        in_space = input_spaces["preprocessing_inputs"]  # type: Space
        # A list of all axes over which to normalize (exclude batch rank).
        self.axes = list(range(1 if in_space.has_batch_rank else 0, len(in_space.get_shape(with_batch_rank=False))))

    @rlgraph_api
    def _graph_fn_apply(self, preprocessing_inputs):
        min_value = preprocessing_inputs
        max_value = preprocessing_inputs

        if get_backend() == "tf":
            import tensorflow as tf
            # Iteratively reduce dimensionality across all axes to get the min/max values for each sample in the batch.
            for axis in self.axes:
                min_value = tf.reduce_min(input_tensor=min_value, axis=axis, keep_dims=True)
                max_value = tf.reduce_max(input_tensor=max_value, axis=axis, keep_dims=True)

        # Add some small constant to never let the range be zero.
        return (preprocessing_inputs - min_value) / (max_value - min_value + SMALL_NUMBER)

