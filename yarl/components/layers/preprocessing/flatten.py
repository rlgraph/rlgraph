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

from yarl import YARLError
from yarl.utils.util import get_shape
from yarl.spaces import Space, IntBox

from yarl.components.layers.preprocessing import PreprocessLayer


class Flatten(PreprocessLayer):
    """
    Flattens the input by reshaping it, excluding the batch-rank (if there is one).
    e.g. input FloatBox(shape=(None, 2, 3, 4)) -> flatten -> FloatBox(shape=(None, 24))

    If the input is an IntBox, will (optionally) flatten for categories as well.
    e.g. input Space=IntBox(4) -> flatten -> FloatBox(shape=(4,)).
    """

    def __init__(self, flatten_categories=True, scope="flatten", **kwargs):
        """
        Args:
            flatten_categories (bool): Whether to flatten also IntBox categories. Default: True.
        """
        super(Flatten, self).__init__(scope=scope, **kwargs)

        self.has_batch = None

        self.flatten_categories = flatten_categories
        self.num_categories = 1

    def check_input_spaces(self, input_spaces):
        # Check whether our input space has-batch or not and store this information here.
        in_space = input_spaces["input"]  # type: Space
        self.has_batch = in_space.has_batch_rank
        # Check whether we can flatten the incoming categories of an IntBox into a FloatBox with additional
        # rank (categories rank).
        if isinstance(in_space, IntBox) and self.flatten_categories is True:
            if in_space.num_categories is False:
                raise YARLError("ERROR: Cannot flatten categories if incoming space ({}) does not have global "
                                "bounds!".format(in_space))
            else:
                self.num_categories = in_space.num_categories

    def _graph_fn_apply(self, input_):
        if self.has_batch:
            shape = (-1, np.prod(get_shape(input_)[1:]) * self.num_categories)
        else:
            shape = tuple([get_shape(input_, flat=True) * self.num_categories])

        return tf.reshape(tensor=input_, shape=shape)
