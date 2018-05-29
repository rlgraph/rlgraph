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

import tensorflow as tf

from yarl.utils.util import get_rank
from .preprocess_layer import PreprocessLayer


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

    def _graph_fn_apply(self, images):
        """
        Images come in with either a batch dimension or not.
        However, this
        """
        # Do some sanity checking.
        rank = get_rank(images)
        assert rank == 3 or rank == 4, "ERROR: Given image's rank ({}) is not 3 or 4!".format(rank)

        return tf.image.resize_images(images=images, size=(self.width, self.height))

