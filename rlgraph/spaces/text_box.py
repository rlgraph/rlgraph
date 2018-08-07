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

from rlgraph.spaces.box_space import BoxSpace


class TextBox(BoxSpace):
    """
    A text box in TXT^n where the shape means the number of text chunks in each dimension.
    """

    def __init__(self, shape=(), add_batch_rank=False, add_time_rank=False):
        """
        Args:
            shape (tuple): The shape of this space.
        """
        # TODO: low/high got to be handled somehow (not needed for text spaces).
        super(TextBox, self).__init__(low=0, high=0, add_batch_rank=add_batch_rank, add_time_rank=add_time_rank)

        self._dtype = np.unicode_

        assert isinstance(shape, tuple), "ERROR: `shape` must be a tuple."
        self._shape = shape

    def sample(self, size=None):
        # TODO: Make it such that it doesn't only produce number strings.
        shape = self._get_np_shape(num_samples=size)
        sample_ = np.random.randint(low=32, high=127, size=shape)
        if shape == () or shape is None:
            return str(sample_)
        else:
            return sample_.astype(self.dtype)

    def contains(self, sample):
        sample_shape = sample.shape if not isinstance(sample, str) else ()
        return sample_shape == self.shape
