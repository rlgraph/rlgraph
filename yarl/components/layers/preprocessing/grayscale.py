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

from yarl.utils.util import get_rank, get_shape
from .preprocess_layer import PreprocessLayer


class GrayScale(PreprocessLayer):
    """
    A simple grayscale converter for RGB images of arbitrary dimensions (normally, an image is 2D).
    """
    def __init__(self, weights=None, keep_rank=False, scope="grayscale", **kwargs):
        """
        Args:
            weights (Optional[tuple,list]): A list/tuple of three items indicating the weights to apply to the 3 color
                channels (RGB).
            keep_rank (bool): Whether to keep the color-depth rank in the pre-processed tensor (default: False).
        """
        super(GrayScale, self).__init__(deterministic=True, scope=scope, **kwargs)
        self.weights = weights or (0.299, 0.587, 0.114)
        self.keep_rank = keep_rank

    def _computation_apply(self, image, key):
        """
        Gray-scales an incoming image of arbitrary rank (normally rank=3, width/height/colors).
        However, the last rank must be of size 3 (RGB color images).

        Args:
            image (tensor): A single image to be gray-scaled (may be nD + 3 (required!) colors).

        Returns:
            op: The op for processing the image.
        """
        assert get_shape(image)[-1] == 3, "ERROR: Given image is not RGB (last rank must be 3)!"
        shape = tuple(1 for _ in range(get_rank(image) - 1)) + (3,)
        weights_reshaped = np.reshape(a=self.weights, newshape=shape)
        return tf.reduce_sum(weights_reshaped * image, axis=-1, keepdims=self.keep_rank)

