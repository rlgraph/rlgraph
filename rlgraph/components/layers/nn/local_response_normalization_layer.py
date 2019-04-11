# Copyright 2018/2019 The RLgraph authors. All Rights Reserved.
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
from rlgraph.components.layers.nn.nn_layer import NNLayer
from rlgraph.utils.decorators import rlgraph_api

if get_backend() == "tf":
    import tensorflow as tf
elif get_backend() == "pytorch":
    import torch.nn as nn


class LocalResponseNormalizationLayer(NNLayer):
    """
    A max-pooling 2D layer.
    """
    def __init__(self, depth_radius=5, bias=1, alpha=1, beta=0.5, **kwargs):
        """
        Args:
            pool_size (Optional[int,Tuple[int,int]]): An int or tuple of 2 ints (height x width) specifying the
                size of the pooling window. Use a  single integer to specify the same value for all spatial dimensions.
            strides (Union[int,Tuple[int]]): Kernel stride size along height and width axis (or one value
                for both directions).
            padding (str): One of 'valid' or 'same'. Default: 'valid'.
            data_format (str): One of 'channels_last' (default) or 'channels_first'. Specifies which rank (first or
                last) is the color-channel. If the input Space is with batch, the batch always has the first rank.
        """
        super(LocalResponseNormalizationLayer, self).__init__(scope=kwargs.pop("scope", "maxpool-2d"), **kwargs)

        self.depth_radius = depth_radius
        self.bias = bias
        self.alpha = alpha
        self.beta = beta

        if get_backend() == "pytorch":
            # Todo: Ensure channels_first?
            self.layer = nn.LocalReponseNorm(
                size=self.depth_radius*2,  # The PyTorch implementation divides the size by 2
                alpha=self.alpha,
                beta=self.beta,
                k=self.bias
            )

    @rlgraph_api
    def _graph_fn_call(self, *inputs):
        if get_backend() == "tf":
            result = tf.nn.local_response_normalization(
                inputs[0], depth_radius=self.depth_radius, bias=self.bias, alpha=self.alpha, beta=self.beta
            )
            # TODO: Move into util function.
            if hasattr(inputs[0], "_batch_rank"):
                result._batch_rank = inputs[0]._batch_rank
            if hasattr(inputs[0], "_time_rank"):
                result._time_rank = inputs[0]._time_rank
            return result
