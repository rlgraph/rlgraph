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

from .space import Space
import numpy as np
from cached_property import cached_property


class Continuous(Space):
    """
    A box in R^n (each coordinate is bounded).
    """

    def __init__(self, low=None, high=None, shape=None):
        """
        Args:
            low (any): The lower bound (see Valid Inputs for more information).
            high (any): The upper bound (see Valid Inputs for more information).
            shape (tuple): The shape of this space.

        Valid inputs:
            Continuous(0.0, 1.0) # low and high are given as scalars and shape is assumed to be ()
                -> single scalar between low and high.
            Continuous(-1.0, 1.0, (3,4)) # low and high are scalars, and shape is provided -> nD array
                where all(!) elements are between low and high.
            Continuous(np.array([-1.0,-2.0]), np.array([2.0,4.0])) # low and high are arrays of the same shape
                (no shape given!) -> nD array where each dimension has different bounds.

            NOT SUPPORTED ATM:
                Continuous(None, None, (2,3,4)) # low and high are not given (unknown) -> figure out bounds by
                    observing incoming samples (use flatten(_batch)? and unflatten(_batch)? methods to get samples).
        """
        self.is_scalar = False  # whether we are a single scalar (shape=())
        self.has_unknown_bounds = False
        self.has_flex_bounds = False

        # Single float (may be bounded)
        if shape is None:
            if isinstance(low, (int, float)) and isinstance(high, (int, float)):
                assert low < high
                self.low = float(low)
                self.high = float(high)
                self.is_scalar = True
            elif low is None:
                assert high is None
                self.has_unknown_bounds = True
                self.has_flex_bounds = True
                self.is_scalar = True
                self.low = float("inf")
                self.high = float("-inf")
            else:
                self.low = np.array(low)
                self.high = np.array(high)
                assert self.low.shape == self.high.shape
        # A box (R^n) (may be bounded).
        else:
            if low is None:
                assert high is None
                self.has_unknown_bounds = True
                self.has_flex_bounds = True
                self.low = np.zeros(shape)
                self.high = np.zeros(shape)
            else:
                assert np.isscalar(low) and np.isscalar(high)
                self.low = low + np.zeros(shape)
                self.high = high + np.zeros(shape)

    @cached_property
    def shape(self):
        if self.is_scalar:
            return tuple()
        return self.low.shape

    @property
    def flat_dim(self):
        return int(np.prod(self.shape))

    @property
    def dtype(self):
        return "float32"

    @property
    def bounds(self):
        return self.low, self.high

    def __repr__(self):
        return "Continuous" + str(self.shape)

    def __eq__(self, other):
        return isinstance(other, Continuous) and np.allclose(self.low, other.low) and np.allclose(self.high, other.high)

    def sample(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        # No bounds are known: Pretend we are between 0.0 and 1.0.
        if self.has_unknown_bounds:
            return np.random.uniform(size=None if self.is_scalar else self.low.shape)
        return np.random.uniform(low=self.low, high=self.high, size=None if self.is_scalar else self.low.shape)

    def contains(self, x):
        if self.is_scalar:
            if self.has_unknown_bounds:
                return isinstance(x, (float, int))
            else:
                return self.low <= x <= self.high
        else:
            if x.shape != self.shape:
                return False
            elif self.has_unknown_bounds:
                return True
            return (x >= self.low).all() and (x <= self.high).all()
