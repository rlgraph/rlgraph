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

from .continuous import Continuous
import numpy as np


class IntBox(Continuous):
    """
    A box in Z^n (only integers; each coordinate is bounded)
    e.g. an image (w x h x RGB) where each color channel pixel can be between 0 and 255.
    """
    def __init__(self, low=None, high=None, shape=None, add_batch_rank=False):
        """
        Three kinds of valid input:
            IntBox(0, 1) # low and high are given as scalars and shape is assumed to be ()
            IntBox(-1, 1, (3,4)) # low and high are scalars, and shape is provided
            IntBox(np.array([-1,-2]), np.array([2,4])) # low and high are arrays of the same shape (no shape given!)
        """
        if not isinstance(low, int) or not isinstance(high, int):
            assert low is None or low.shape == high.shape
        super(IntBox, self).__init__(low, high, shape, add_batch_rank=add_batch_rank)

    @property
    def dtype(self):
        return "uint8"

    def __eq__(self, other):
        return isinstance(other, IntBox) and np.allclose(self.low, other.low) and np.allclose(self.high, other.high)

    def sample(self, size=None, seed=None):
        if self.has_unknown_bounds:
            raise RuntimeError("Cannot generate samples of a Space with unknown bounds. "
                               "Need flatten/unflatten samples first!")
        if seed is not None:
            np.random.seed(seed)
        return np.random.randint(low=self.low - 1,high=self.high + 1, size=size)

    def contains(self, sample):
        # Check for int type in given sample.
        if not np.equal(np.mod(sample, 1), 0).all():
            # Wrong type.
            return False
        # Let parent handle it.
        return super(IntBox, self).contains(sample)
