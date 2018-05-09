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

import numpy as np
import tensorflow as tf

from yarl.utils.util import get_rank
from .preprocess_layer import PreprocessLayer


class GrayScale(PreprocessLayer):
    """
    A simple grayscale converter for RGB images.
    """
    def __init__(self, weights=None, keep_rank=False, **kwargs):
        """
        Args:
            weights (Optional[tuple,list]): A list/tuple of three items indicating the weights to apply to the 3 color
                channels (RGB).
            keep_rank (bool): Whether to keep the color-depth rank in the pre-processed tensor (default: False).
        """
        self.weights = weights or (0.299, 0.587, 0.114)
        self.weights_reshaped = None
        self.keep_rank = keep_rank
        super(GrayScale, self).__init__(**kwargs)

    def shape_test(self, input_shape):
        self.weights_reshaped = np.reshape(a=self.weights, newshape=tuple((1 for _ in range(input_shape))) + (3,))

    def preprocess(self, input_):
        return tf.reduce_sum(self.weights_reshaped * input_, axis=-1, keepdims=self.keep_rank)

