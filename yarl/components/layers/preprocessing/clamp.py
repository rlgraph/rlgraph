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

from .preprocess_layer import PreprocessLayer


class Clamp(PreprocessLayer):
    """
    A simple clamp layer. Clamps each value in the input tensor between min_ and max_.
    """
    def __init__(self,  min_, max_, scope="clamp", **kwargs):
        """
        Args:
            min_ (float): The min value that any value in the inputs may have.
            max_ (float): The max value that any value in the inputs may have.
        """
        super(Clamp, self).__init__(scope=scope, **kwargs)
        self.min_ = min_
        self.max_ = max_

    def _computation_apply(self, input_):
        return tf.clip_by_value(t=input_, clip_value_min=self.min_, clip_value_max=self.max_)
