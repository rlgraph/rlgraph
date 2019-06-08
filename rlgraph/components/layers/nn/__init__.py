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

from rlgraph.components.layers.nn.concat_layer import ConcatLayer
from rlgraph.components.layers.nn.conv2d_layer import Conv2DLayer
from rlgraph.components.layers.nn.conv2d_transpose_layer import Conv2DTransposeLayer
from rlgraph.components.layers.nn.dense_layer import DenseLayer
from rlgraph.components.layers.nn.local_response_normalization_layer import LocalResponseNormalizationLayer
from rlgraph.components.layers.nn.lstm_layer import LSTMLayer
from rlgraph.components.layers.nn.maxpool2d_layer import MaxPool2DLayer
from rlgraph.components.layers.nn.multi_lstm_layer import MultiLSTMLayer
from rlgraph.components.layers.nn.nn_layer import NNLayer
from rlgraph.components.layers.nn.residual_layer import ResidualLayer

NNLayer.__lookup_classes__ = dict(
    concat=ConcatLayer,
    concatlayer=ConcatLayer,
    conv2d=Conv2DLayer,
    conv2dlayer=Conv2DLayer,
    conv2dtranspose=Conv2DTransposeLayer,
    conv2dtransposelayer=Conv2DTransposeLayer,
    dense=DenseLayer,
    denselayer=DenseLayer,
    fc=DenseLayer,
    fclayer=DenseLayer,
    lstm=LSTMLayer,
    lstmlayer=LSTMLayer,
    maxpool2d=MaxPool2DLayer,
    maxpool2dlayer=MaxPool2DLayer,
    multilstm=MultiLSTMLayer,
    multilstmlayer=MultiLSTMLayer,
    residual=ResidualLayer,
    residuallayer=ResidualLayer,
    localresponsenormalization=LocalResponseNormalizationLayer,
    localresponsenormalizationlayer=LocalResponseNormalizationLayer
)

__all__ = ["NNLayer"] + list(set(map(lambda x: x.__name__, NNLayer.__lookup_classes__.values())))
