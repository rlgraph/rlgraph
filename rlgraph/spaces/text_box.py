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

import numpy as np
# import random
# import string

from rlgraph.spaces.box_space import BoxSpace


class TextBox(BoxSpace):
    """
    A text box in TXT^n where the shape means the number of text chunks in each dimension.
    """

    def __init__(self, shape=(), **kwargs):
        """
        Args:
            shape (tuple): The shape of this space.
        """
        # Set both low/high to 0 (make no sense for text).
        super(TextBox, self).__init__(low=0, high=0, **kwargs)

        # Set dtype to numpy's unicode type.
        self.dtype = np.unicode_

        assert isinstance(shape, tuple), "ERROR: `shape` must be a tuple."
        self._shape = shape

    def sample(self, size=None, fill_value=None):
        shape = self._get_np_shape(num_samples=size)

        # TODO: Make it such that it doesn't only produce number strings (using `petname` module?).
        sample_ = np.full(shape=shape, fill_value=fill_value, dtype=self.dtype)

        return sample_.astype(self.dtype)

    def contains(self, sample):
        sample_shape = sample.shape if not isinstance(sample, str) else ()
        return sample_shape == self.shape
