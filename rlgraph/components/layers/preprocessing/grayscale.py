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

import cv2
import numpy as np
from six.moves import xrange as range_

from rlgraph import get_backend
from rlgraph.components.layers.preprocessing import PreprocessLayer
from rlgraph.utils.decorators import rlgraph_api
from rlgraph.utils.ops import flatten_op, unflatten_op
from rlgraph.utils.util import get_rank, get_shape, dtype as dtype_

if get_backend() == "tf":
    import tensorflow as tf
elif get_backend() == "pytorch":
    import torch


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

    def create_variables(self, input_spaces, action_space=None):
        in_space = input_spaces["preprocessing_inputs"]
        self.output_spaces = flatten_op(self.get_preprocessed_space(in_space))

    @rlgraph_api(flatten_ops=True, split_ops=True)
    def _graph_fn_apply(self, preprocessing_inputs):
        """
        Gray-scales images of arbitrary rank.
        Normally, the images' rank is 3 (width/height/colors), but can also be: batch/width/height/colors, or any other.
        However, the last rank must be of size: len(self.weights).

        Args:
            preprocessing_inputs (tensor): Single image or a batch of images to be gray-scaled (last rank=n colors, where
                n=len(self.weights)).

        Returns:
            DataOp: The op for processing the images.
        """
        # The reshaped weights used for the grayscale operation.
        if isinstance(preprocessing_inputs, list):
            preprocessing_inputs = np.asarray(preprocessing_inputs)
        images_shape = get_shape(preprocessing_inputs)
        assert images_shape[-1] == self.last_rank,\
            "ERROR: Given image's shape ({}) does not match number of weights (last rank must be {})!".\
            format(images_shape, self.last_rank)
        if self.backend == "python" or get_backend() == "python":
            if preprocessing_inputs.ndim == 4:
                grayscaled = []
                for i in range_(len(preprocessing_inputs)):
                    scaled = cv2.cvtColor(preprocessing_inputs[i], cv2.COLOR_RGB2GRAY)
                    grayscaled.append(scaled)
                scaled_images = np.asarray(grayscaled)

                # Keep last dim.
                if self.keep_rank:
                    scaled_images = scaled_images[:, :, :, np.newaxis]
            else:
                # Sample by sample.
                scaled_images = cv2.cvtColor(preprocessing_inputs, cv2.COLOR_RGB2GRAY)

            return scaled_images
        elif get_backend() == "pytorch":
            if len(preprocessing_inputs.shape) == 4:
                grayscaled = []
                for i in range_(len(preprocessing_inputs)):
                    scaled = cv2.cvtColor(preprocessing_inputs[i].numpy(), cv2.COLOR_RGB2GRAY)
                    grayscaled.append(scaled)
                scaled_images = np.asarray(grayscaled)
                # Keep last dim.
                if self.keep_rank:
                    scaled_images = scaled_images[:, :, :, np.newaxis]
            else:
                # Sample by sample.
                scaled_images = cv2.cvtColor(preprocessing_inputs.numpy(), cv2.COLOR_RGB2GRAY)
            return torch.tensor(scaled_images)
        elif get_backend() == "tf":
            weights_reshaped = np.reshape(
                self.weights, newshape=tuple([1] * (get_rank(preprocessing_inputs) - 1)) + (self.last_rank,)
            )

            # Do we need to convert?
            # The dangerous thing is that multiplying an int tensor (image) with float weights results in an all
            # 0 tensor).
            if "int" in str(dtype_(preprocessing_inputs.dtype)):
                weighted = weights_reshaped * tf.cast(preprocessing_inputs, dtype=dtype_("float"))
            else:
                weighted = weights_reshaped * preprocessing_inputs

            reduced = tf.reduce_sum(weighted, axis=-1, keepdims=self.keep_rank)

            # Cast back to original dtype.
            if "int" in str(dtype_(preprocessing_inputs.dtype)):
                reduced = tf.cast(reduced, dtype=preprocessing_inputs.dtype)

            return reduced
