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

from rlgraph import get_backend
from rlgraph.utils.ops import flatten_op, unflatten_op
from rlgraph.components.layers.preprocessing import PreprocessLayer
from rlgraph.utils.decorators import rlgraph_api


if get_backend() == "tf":
    import tensorflow as tf
elif get_backend() == "pytorch":
    import torch


class ImageCrop(PreprocessLayer):
    """
    Crops one or more images to a new size without touching the color channel.
    """
    def __init__(self, x=0, y=0, width=0, height=0, scope="image-crop", **kwargs):
        """
        Args:
            x (int): Start x coordinate.
            y (int): Start y coordinate.
            width (int): Width of resulting image.
            height (int): Height of resulting image.
        """
        super(ImageCrop, self).__init__(scope=scope, **kwargs)
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        assert self.x >= 0
        assert self.y >= 0
        assert self.width > 0
        assert self.height > 0

        # The output spaces after preprocessing (per flat-key).
        self.output_spaces = dict()

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

    def create_variables(self, input_spaces, action_space=None):
        in_space = input_spaces["preprocessing_inputs"]

        self.output_spaces = flatten_op(self.get_preprocessed_space(in_space))

    @rlgraph_api(flatten_ops=True, split_ops=True, add_auto_key_as_first_param=True)
    def _graph_fn_apply(self, key, preprocessing_inputs):
        """
        Images come in with either a batch dimension or not.
        """
        if self.backend == "python" or get_backend() == "python":
            if isinstance(preprocessing_inputs, list):
                preprocessing_inputs = np.asarray(preprocessing_inputs)
            # Preserve batch dimension.
            if self.output_spaces[key].has_batch_rank is True:
                return preprocessing_inputs[:, self.y:self.y + self.height, self.x:self.x + self.width]
            else:
                return preprocessing_inputs[self.y:self.y + self.height, self.x:self.x + self.width]
        elif get_backend() == "pytorch":
            if isinstance(preprocessing_inputs, list):
                preprocessing_inputs = torch.tensor(preprocessing_inputs)

            # TODO: the reason this key check is there is due to call during meta graph build - > out spaces
            # do not exist yet  -> need better solution.
            # Preserve batch dimension.
            if key in self.output_spaces and self.output_spaces[key].has_batch_rank is True:
                return preprocessing_inputs[:, self.y:self.y + self.height, self.x:self.x + self.width]
            else:
                return preprocessing_inputs[self.y:self.y + self.height, self.x:self.x + self.width]
        elif get_backend() == "tf":
            return tf.image.crop_to_bounding_box(
                image=preprocessing_inputs,
                offset_height=self.y,
                offset_width=self.x,
                target_height=self.height,
                target_width=self.width
            )
