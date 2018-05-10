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
        self.keep_rank = keep_rank
        super(GrayScale, self).__init__(**kwargs)

    def _precomputation_apply(self, *inputs):
        """
        Grayscales an incoming image of arbitrary rank (normally rank=3, width/height/colors).
        `weights_reshaped` must already match the rank of this image

        Args:
            inputs (any): The input Sockets' ops.

        Returns:
            The arguments that need to be passed into
        """
        # Sanity checks.
        assert len(inputs) == 1, "ERROR: GrayScale can only take one input Socket!"
        image = inputs[0]
        assert image.get_shape
        shape = tuple(1 for _ in range(get_rank(image) - 1)) + (3,)
        return image, np.reshape(a=self.weights, newshape=shape)

    def _computation_apply(self, image, weights_reshaped):
        return tf.reduce_sum(weights_reshaped * image, axis=-1, keepdims=self.keep_rank)
