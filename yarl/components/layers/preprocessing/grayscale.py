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
        super(GrayScale, self).__init__(scope=scope, **kwargs)
        self.weights = weights or (0.299, 0.587, 0.114)  # magic RGB-weights for "natural" gray-scaling results
        self.last_rank = len(self.weights)
        self.keep_rank = keep_rank

    def _computation_apply(self, images):
        """
        Gray-scales images of arbitrary rank.
        Normally, the images' rank is 3 (width/height/colors), but can also be: batch/width/height/colors, or any other.
        However, the last rank must be of size: len(self.weights).

        Args:
            images (tensor): Single image or a batch of images to be gray-scaled (last rank=n colors, where
                m=len(self.weights)).

        Returns:
            op: The op for processing the images.
        """
        images_shape = get_shape(images)
        assert images_shape[-1] == self.last_rank, "ERROR: Given image's shape ({}) does not match number of " \
                                                   "weights (last rank must be {})!".format(images_shape,
                                                                                            self.last_rank)
        weights_reshaped = np.reshape(a=self.weights,
                                      newshape=tuple([1] * (get_rank(images)-1)) + (self.last_rank,))
        return tf.reduce_sum(input_tensor=weights_reshaped * images, axis=-1, keepdims=self.keep_rank)

