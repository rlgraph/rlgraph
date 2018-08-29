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
cv2.ocl.setUseOpenCL(False)
import numpy as np
from six.moves import xrange as range_

from rlgraph import get_backend
from rlgraph.utils import RLGraphError
from rlgraph.utils.ops import unflatten_op
from rlgraph.components.layers.preprocessing import PreprocessLayer

if get_backend() == "tf":
    import tensorflow as tf
    from tensorflow.python.ops.image_ops_impl import ResizeMethod


class ImageResize(PreprocessLayer):
    """
    Resizes one or more images to a new size without touching the color channel.
    """
    def __init__(self, width, height, interpolation="area", scope="image-resize", **kwargs):
        """
        Args:
            width (int): The new width.
            height (int): The new height.
            interpolation (str): One of "bilinear", "area". Default: "bilinear" (which is also the default for both
                cv2 and tf).
        """
        super(ImageResize, self).__init__(scope=scope, **kwargs)
        self.width = width
        self.height = height
        
        if interpolation == "bilinear":
            self.cv2_interpolation = cv2.INTER_LINEAR
            self.tf_interpolation = ResizeMethod.BILINEAR
        elif interpolation == "area":
            self.cv2_interpolation = cv2.INTER_AREA
            self.tf_interpolation = ResizeMethod.AREA
        else:
            raise RLGraphError("Invalid interpolation algorithm {}!. Allowed are 'bilinear' and "
                               "'area'.".format(interpolation))

        # The output spaces after preprocessing (per flat-key).
        self.output_spaces = None

    def get_preprocessed_space(self, space):
        ## Test sending np samples to get number of return values and output spaces without having to call
        ## the tf graph_fn.
        #backend = self.backend
        #self.backend = "python"
        #sample = space.sample(size=1)
        #out = self._graph_fn_apply(sample)
        #new_space = get_space_from_op(out)
        #self.backend = backend
        #return new_space

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

    def create_variables(self, input_spaces, action_space=None):
        in_space = input_spaces["preprocessing_inputs"]
        self.output_spaces = self.get_preprocessed_space(in_space)

    # @rlgraph.graph_fn
    def _graph_fn_apply(self, preprocessing_inputs):
        """
        Images come in with either a batch dimension or not.
        """
        if self.backend == "python" or get_backend() == "python":
            if isinstance(preprocessing_inputs, list):
                preprocessing_inputs = np.asarray(preprocessing_inputs)
            had_single_color_dim = (preprocessing_inputs.shape[-1] == 1)
            # Batch of samples.
            if preprocessing_inputs.ndim == 4:
                resized = []
                for i in range_(len(preprocessing_inputs)):
                    resized.append(cv2.resize(
                        preprocessing_inputs[i], dsize=(self.width, self.height), interpolation=self.cv2_interpolation)
                    )
                resized = np.asarray(resized)
            # Single sample.
            else:
                resized = cv2.resize(
                    preprocessing_inputs, dsize=(self.width, self.height), interpolation=self.cv2_interpolation
                )

            # cv2.resize removes the color rank, if its dimension is 1 (e.g. grayscale), add it back here.
            if had_single_color_dim is True:
                resized = np.expand_dims(resized, axis=-1)

            return resized

        elif get_backend() == "tf":
            return tf.image.resize_images(
                images=preprocessing_inputs, size=(self.width, self.height), method=self.tf_interpolation
            )

