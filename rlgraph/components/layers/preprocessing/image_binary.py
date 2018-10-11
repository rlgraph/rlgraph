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
from rlgraph.components.layers.preprocessing import PreprocessLayer
from rlgraph.utils.decorators import rlgraph_api

if get_backend() == "tf":
    import tensorflow as tf


class ImageBinary(PreprocessLayer):
    """
    # TODO: Better to move this into grayscale! When needed.
    A simple binary converter for images of arbitrary dimensions. All non-black pixels are converted to
    1.0s, all black pixels (all 0.0 in last rank) remain.
    """
    def __init__(self, threshold=0.0, keep_rank=False, scope="image-binary", **kwargs):
        """
        Args:
            keep_rank (bool): Whether to keep the color-depth rank in the pre-processed tensor (default: False).
        """
        super(ImageBinary, self).__init__(scope=scope, **kwargs)

        # The threshold after which the sum of
        # The dimension of the last rank (e.g. color rank of the image).
        self.last_rank = None
        # Whether to keep the last rank with dim=1.
        self.keep_rank = keep_rank

    def check_input_spaces(self, input_spaces, action_space=None):
        in_space = input_spaces["preprocessing_inputs"]
        self.last_rank = in_space.shape[-1]

    @rlgraph_api
    def _graph_fn_apply(self, preprocessing_inputs):
        """
        Converts the images into binary images by replacing all non-black (at least one channel value is not 0.0)
        to 1.0 and leaves all black pixels (all channel values 0.0) as-is.

        Args:
            preprocessing_inputs (tensor): Single image or a batch of images to be converted into a binary image (last rank=n colors,
                where n=len(self.weights)).

        Returns:
            DataOp: The op for processing the images.
        """
        if get_backend() == "tf":
            # Sum over the color channel.
            color_channel_sum = tf.reduce_sum(input_tensor=preprocessing_inputs, axis=-1, keepdims=self.keep_rank)
            # Reduce the image to only 0.0 or 1.0.
            binary_image = tf.where(
                tf.greater(color_channel_sum, 0.0), tf.ones_like(color_channel_sum),
                tf.zeros_like(color_channel_sum)
            )
            return binary_image

