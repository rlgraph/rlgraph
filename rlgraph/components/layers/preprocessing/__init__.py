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
from rlgraph.components.layers.preprocessing.flatten import Flatten
from rlgraph.components.layers.preprocessing.grayscale import GrayScale
from rlgraph.components.layers.preprocessing.image_binary import ImageBinary
from rlgraph.components.layers.preprocessing.image_resize import ImageResize
from rlgraph.components.layers.preprocessing.normalize import Normalize
from rlgraph.components.layers.preprocessing.multiply_divide import Multiply, Divide
from rlgraph.components.layers.preprocessing.sequence import Sequence

PreprocessLayer.__lookup_classes__ = dict(
    clip=Clip,
    divide=Divide,
    flatten=Flatten,
    grayscale=GrayScale,
    imagebinary=ImageBinary,
    imageresize=ImageResize,
    multiply=Multiply,
    normalize=Normalize,
    sequence=Sequence
)

__all__ = ["PreprocessLayer",
           "Clip", "Divide", "Flatten", "GrayScale", "ImageBinary", "ImageResize", "Multiply", "Normalize", "Sequence"]


