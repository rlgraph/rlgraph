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

from rlgraph.components.layers.preprocessing.preprocess_layer import PreprocessLayer

from rlgraph.components.layers.preprocessing.clip import Clip
from rlgraph.components.layers.preprocessing.concat import Concat
from rlgraph.components.layers.preprocessing.grayscale import GrayScale
from rlgraph.components.layers.preprocessing.image_binary import ImageBinary
from rlgraph.components.layers.preprocessing.convert_type import ConvertType
from rlgraph.components.layers.preprocessing.image_crop import ImageCrop
from rlgraph.components.layers.preprocessing.image_resize import ImageResize
from rlgraph.components.layers.preprocessing.normalize import Normalize
from rlgraph.components.layers.preprocessing.multiply_divide import Multiply, Divide
from rlgraph.components.layers.preprocessing.reshape import ReShape
from rlgraph.components.layers.preprocessing.sequence import Sequence
from rlgraph.components.layers.preprocessing.transpose import Transpose

PreprocessLayer.__lookup_classes__ = dict(
    clip=Clip,
    concat=Concat,
    divide=Divide,
    grayscale=GrayScale,
    imagebinary=ImageBinary,
    converttype=ConvertType,
    imagecrop=ImageCrop,
    imageresize=ImageResize,
    multiply=Multiply,
    normalize=Normalize,
    reshape=ReShape,
    sequence=Sequence,
    transpose=Transpose,
)


__all__ = ["PreprocessLayer"] + \
          list(set(map(lambda x: x.__name__, PreprocessLayer.__lookup_classes__.values())))
