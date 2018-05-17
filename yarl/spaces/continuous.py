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

import numpy as np
from cached_property import cached_property

from yarl import YARLError
from .space import Space


class Continuous(Space):
    """
    A box in R^n (each coordinate is bounded).
    """

    def __init__(self, low=None, high=None, shape=None, add_batch_rank=False):
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
            Continuous(None, None, (2,3,4)) # low and high are not given (unknown).
        """
        super(Continuous, self).__init__(add_batch_rank=add_batch_rank)
        self.is_scalar = False  # whether we are a single scalar (shape=())
        self.has_unknown_bounds = False
        self.has_flex_bounds = False

        # Single float (may be bounded)
        if shape is None or shape == ():
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
                if shape == ():
                    raise YARLError("ERROR: Shape cannot be () if low and/or high are given as shape-tuples!")
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
            return ()
        return tuple(self.low.shape)

    @cached_property
    def shape_with_batch_rank(self):
        if self.is_scalar:
            return self.batch_rank_tuple
        return tuple(self.batch_rank_tuple + self.shape)

    @cached_property
    def flat_dim(self):
        return int(np.prod(self.shape))

    @cached_property
    def dtype(self):
        return "float32"

    @property
    def bounds(self):
        return self.low, self.high

    #def get_initializer(self, specification):
    #    if backend() == "tf":
    #        return Initializer.from_spec(shape=self.shape, specification=specification)
    #    else:
    #        raise YARLError("ERROR: Pytorch not supported yet!")

    def __repr__(self):
        return "{}({}{})".format(type(self).__name__.title(), self.shape, "; +batch" if self.has_batch_rank else "")

    def __eq__(self, other):
        return isinstance(other, Continuous) and np.allclose(self.low, other.low) and np.allclose(self.high, other.high)

    def sample(self, size=None, seed=None):
        shape = self._get_np_shape(num_samples=size)
        if seed is not None:
            np.random.seed(seed)
        # No bounds are known: Pretend we are between 0.0 and 1.0.
        if self.has_unknown_bounds:
            return np.random.uniform(size=shape)
        return np.random.uniform(low=self.low, high=self.high, size=shape)

    def contains(self, sample):
        if self.is_scalar:
            if self.has_unknown_bounds:
                return isinstance(sample, (float, int))
            else:
                return self.low <= sample <= self.high
        else:
            if sample.shape != self.shape:
                return False
            elif self.has_unknown_bounds:
                return True
            return (sample >= self.low).all() and (sample <= self.high).all()
