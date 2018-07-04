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
        super(Flatten, self).__init__(scope=scope, add_auto_key_as_first_param=True, **kwargs)

        self.has_batch = None

        self.flatten_categories = flatten_categories
        # Stores the number of categories in IntBoxes.
        self.num_categories = dict()

    def check_input_spaces(self, input_spaces, action_space):
        super(Flatten, self).check_input_spaces(input_spaces, action_space)

        # Check whether our input space has-batch or not and store this information here.
        in_space = input_spaces["apply"][0]  # type: Space
        self.has_batch = in_space.has_batch_rank
        # Check whether we have to flatten the incoming categories of an IntBox into a FloatBox with additional
        # rank (categories rank). Store the dimension of this additional rank in the `self.num_categories` dict.
        if self.flatten_categories is True:
            def mapping_func(key, space):
                if isinstance(space, IntBox):
                    # Must have global bounds (bounds valid for all axes).
                    if space.num_categories is False:
                        raise YARLError("ERROR: Cannot flatten categories if one of the IntBox spaces ({}={}) does not "
                                        "have global bounds (its `num_categories` is False)!".format(key, space))
                    return space.num_categories
                # No categories. Keep as is.
                return 1
            self.num_categories = in_space.flatten(mapping=mapping_func)

    def _graph_fn_apply(self, key, input_):
        if self.has_batch:
            shape = (-1, int(np.prod(get_shape(input_)[1:])))
        else:
            shape = tuple([get_shape(input_, flat=True)])

        reshaped = tf.reshape(tensor=input_, shape=shape)
        # Create a one-hot axis for the categories at the end.
        if self.num_categories[key] > 1:
            reshaped = tf.squeeze(tf.one_hot(indices=reshaped, depth=self.num_categories[key], axis=1), axis=2)
        return tf.identity(reshaped, name="flattened")
