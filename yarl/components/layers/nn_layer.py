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

from yarl import backend
from .layer_component import LayerComponent

from functools import partial


class NNLayer(LayerComponent):
    """
    A neural-net layer wrapper class that can incorporate a backend-specific layer object.
    """
    def __init__(self, *sub_components, class_=None, **kwargs):
        """
        Keyword Args:
            class (class): The wrapped tf.layers class to use.
            **kwargs (any): Kwargs to be passed to the native backend's layers's constructor.
        """
        sub_components = list(sub_components)
        assert class_, "ERROR: class_ parameter needs to be given as kwarg in c'tor of NNLayer!"

        super(NNLayer, self).__init__(*sub_components, **kwargs)
        self.class_ = class_
        self.kwargs = kwargs

    def _computation_apply(self, input_):
        """
        Only can make_template from this function after(!) we know what the "output"?? socket's shape will be.
        """
        # TODO: wrap pytorch's torch.nn classes
        if backend() == "tf":
            return self.class_(input_, **self.kwargs)


# Create some fixtures for all layer types for simplicity (w/o the need to add any code).
if backend() == "tf":
    import tensorflow as tf

    DenseLayer = partial(NNLayer, class_=tf.layers.Dense)
    Conv1DLayer = partial(NNLayer,class_=tf.layers.Conv1D)
    Conv2DLayer = partial(NNLayer, class_=tf.layers.Conv2D)
    Conv2DTransposeLayer = partial(NNLayer, class_=tf.layers.Conv2DTranspose)
    Conv3DLayer = partial(NNLayer, class_=tf.layers.Conv3D)
    Conv3DTransposeLayer = partial(NNLayer, class_=tf.layers.Conv3DTranspose)
    AveragePooling1DLayer = partial(NNLayer, class_=tf.layers.AveragePooling1D)
    AveragePooling2DLayer = partial(NNLayer, class_=tf.layers.AveragePooling2D)
    AveragePooling3DLayer = partial(NNLayer, class_=tf.layers.AveragePooling3D)
    BatchNormalizationLayer = partial(NNLayer, class_=tf.layers.BatchNormalization)
    DropoutLayer = partial(NNLayer, class_=tf.layers.Dropout)
    FlattenLayer = partial(NNLayer, class_=tf.layers.Flatten)
    MaxPooling1DLayer = partial(NNLayer, class_=tf.layers.MaxPooling1D)
    MaxPooling2DLayer = partial(NNLayer, class_=tf.layers.MaxPooling2D)
    MaxPooling3DLayer = partial(NNLayer, class_=tf.layers.MaxPooling3D)
