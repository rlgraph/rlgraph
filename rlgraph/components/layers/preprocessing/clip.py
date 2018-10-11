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

import numpy as np

from rlgraph import get_backend
from rlgraph.components.layers.preprocessing import PreprocessLayer
from rlgraph.utils.decorators import rlgraph_api

if get_backend() == "tf":
    import tensorflow as tf


class Clip(PreprocessLayer):
    """
    A simple clip-by-value layer. Clips each value in the input tensor between `min` and `max`.
    """
    def __init__(self, min=0.0, max=1.0, scope="clip", **kwargs):
        """
        Args:
            min\_ (float): The min value that any value in the input can have.
            max\_ (float): The max value that any value in the input can have.
        """
        super(Clip, self).__init__(scope=scope, **kwargs)
        self.min = min
        self.max = max

    @rlgraph_api
    def _graph_fn_apply(self, preprocessing_inputs):
        if self.backend == "python" or get_backend() == "python":
            return np.clip(preprocessing_inputs, a_min=self.min, a_max=self.max)
        elif get_backend() == "tf":
            return tf.clip_by_value(t=preprocessing_inputs, clip_value_min=self.min, clip_value_max=self.max)

