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

from rlgraph.backend_system import get_backend
from rlgraph.components.layers.preprocessing import PreprocessLayer
from rlgraph.utils.ops import flatten_op, unflatten_op
from rlgraph.utils.util import get_rank, get_shape

if get_backend() == "tf":
    import tensorflow as tf


class GrayScale(PreprocessLayer):
    """
    A simple grayscale converter for RGB images of arbitrary dimensions (normally, an image is 2D).

    [1]: C Kanan, GW Cottrell: Color-to-Grayscale: Does the Method Matter in Image Recognition? - PLOS One (2012)
    """
    def __init__(self, weights=None, keep_rank=False, scope="grayscale", **kwargs):
        """
        Args:
            weights (Optional[tuple,list]): A list/tuple of three items indicating the weights to apply to the 3 color
                channels (RGB).
            keep_rank (bool): Whether to keep the color-depth rank in the pre-processed tensor (default: False).
        """
        super(GrayScale, self).__init__(scope=scope, **kwargs)

        # A list of weights used to reduce the last rank (e.g. color rank) of the inputs.
        self.weights = weights or (0.299, 0.587, 0.114)  # magic RGB-weights for "natural" gray-scaling results
        # The dimension of the last rank (e.g. color rank of the image).
        self.last_rank = len(self.weights)
        # Whether to keep the last rank with dim=1.
        self.keep_rank = keep_rank
        # The output spaces after preprocessing (per flat-key).
        self.output_spaces = None

    def get_preprocessed_space(self, space):
        ret = dict()
        for key, value in space.flatten().items():
            shape = list(value.shape)
            if self.keep_rank is True:
                shape[-1] = 1
            else:
                shape.pop(-1)
            ret[key] = value.__class__(shape=tuple(shape), add_batch_rank=value.has_batch_rank)
        return unflatten_op(ret)

    def check_input_spaces(self, input_spaces, action_space):
        super(GrayScale, self).check_input_spaces(input_spaces, action_space)
        in_space = input_spaces["apply"][0]
        self.output_spaces = flatten_op(self.get_preprocessed_space(in_space))

    def _graph_fn_apply(self, images):
        """
        Gray-scales images of arbitrary rank.
        Normally, the images' rank is 3 (width/height/colors), but can also be: batch/width/height/colors, or any other.
        However, the last rank must be of size: len(self.weights).

        Args:
            images (tensor): Single image or a batch of images to be gray-scaled (last rank=n colors, where
                n=len(self.weights)).

        Returns:
            DataOp: The op for processing the images.
        """
        # The reshaped weights used for the grayscale operation.
        images_shape = get_shape(images)
        assert images_shape[-1] == self.last_rank,\
            "ERROR: Given image's shape ({}) does not match number of weights (last rank must be {})!".\
            format(images_shape, self.last_rank)
        weights_reshaped = np.reshape(a=self.weights,
                                      newshape=tuple([1] * (get_rank(images)-1)) + (self.last_rank,))

        if get_backend() == "tf":
            return tf.reduce_sum(input_tensor=weights_reshaped * images, axis=-1, keepdims=self.keep_rank)

