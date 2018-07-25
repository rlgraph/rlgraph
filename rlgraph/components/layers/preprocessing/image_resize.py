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
from rlgraph.utils.ops import flatten_op, unflatten_op
from rlgraph.components.layers.preprocessing import PreprocessLayer


if get_backend() == "tf":
    import tensorflow as tf


class ImageResize(PreprocessLayer):
    """
    Resizes one or more images to a new size without touching the color channel.
    """
    def __init__(self, width, height, scope="image-resize", **kwargs):
        """
        Args:
            width (int): The new width.
            height (int): The new height.
        """
        super(ImageResize, self).__init__(scope=scope, **kwargs)
        self.width = width
        self.height = height
        # The output spaces after preprocessing (per flat-key).
        self.output_spaces = None

    def get_preprocessed_space(self, space):
        ret = dict()
        for key, value in space.flatten().items():
            # Do some sanity checking.
            rank = value.rank
            assert rank == 2 or rank == 3, \
                "ERROR: Given image's rank (which is {}{}, not counting batch rank) must be either 2 or 3!".\
                format(rank, ("" if key == "" else " for key '{}'".format(key)))
            # Determine the output shape.
            shape = list(value.shape)
            shape[0] = self.width
            shape[1] = self.height
            ret[key] = value.__class__(shape=tuple(shape), add_batch_rank=value.has_batch_rank)
        return unflatten_op(ret)

    def check_input_spaces(self, input_spaces, action_space):
        super(ImageResize, self).check_input_spaces(input_spaces, action_space)
        in_space = input_spaces["apply"][0]

        self.output_spaces = self.get_preprocessed_space(in_space)

    def _graph_fn_apply(self, images):
        """
        Images come in with either a batch dimension or not.
        However, this
        """
        if self.backend == "python" or get_backend() == "python":
            import cv2
            return cv2.resize(images, dsize=(self.width, self.height))  # interpolation=cv2.BILINEAR
        elif get_backend() == "tf":
            return tf.image.resize_images(images=images, size=(self.width, self.height))

