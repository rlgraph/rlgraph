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
import numpy as np

from yarl.utils.util import get_shape
from yarl.spaces import Space

from .preprocess_layer import PreprocessLayer


class Flatten(PreprocessLayer):
    """
    Flattens the input by reshaping it, excluding the batch-rank (if there is one).
    """

    def __init__(self, scope="flatten", **kwargs):
        super(Flatten, self).__init__(scope=scope, **kwargs)
        self.has_batch = None

    def create_variables(self, input_spaces):
        # Check whether our input space has-batch or not and store this information here.
        in_space = input_spaces["input"]  # type: Space
        self.has_batch = in_space.has_batch_rank

    def _graph_fn_apply(self, input_):
        # TODO: Create GraphFunction option to pass input_'s Space along with input_ into these methods.
        shape = (-1, np.prod(get_shape(input_)[1:])) if self.has_batch else tuple([get_shape(input_, flat=True)])
        return tf.reshape(tensor=input_, shape=shape)
