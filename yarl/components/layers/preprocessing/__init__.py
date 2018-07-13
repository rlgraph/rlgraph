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

from yarl.components.layers.preprocessing.preprocess_layer import PreprocessLayer

from yarl.components.layers.preprocessing.clamp import Clamp
from yarl.components.layers.preprocessing.flatten import Flatten
from yarl.components.layers.preprocessing.grayscale import GrayScale
from yarl.components.layers.preprocessing.image_binary import ImageBinary
from yarl.components.layers.preprocessing.image_resize import ImageResize
from yarl.components.layers.preprocessing.normalize import Normalize
from yarl.components.layers.preprocessing.scale import Scale
from yarl.components.layers.preprocessing.sequence import Sequence

PreprocessLayer.__lookup_classes__ = dict(
    clamp=Clamp,
    flatten=Flatten,
    grayscale=GrayScale,
    imagebinary=ImageBinary,
    imageresize=ImageResize,
    normalize=Normalize,
    scale=Scale,
    sequence=Sequence
)

__all__ = ["PreprocessLayer",
           "Clamp", "Flatten", "GrayScale", "ImageBinary", "ImageResize", "Normalize", "Scale", "Sequence"]

