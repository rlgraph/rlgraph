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
from six.moves import xrange as range_
import re

from rlgraph import get_backend
from rlgraph.utils.util import dtype
from rlgraph.utils.initializer import Initializer
from rlgraph.spaces import Space

if get_backend() == "pytorch":
    import torch


class BoxSpace(Space):
    """
    A box in R^n with a shape tuple of len n. Each dimension may be bounded.
    """

    def __init__(self, low, high, shape=None, add_batch_rank=False, add_time_rank=False, time_major=False,
                 dtype=np.float32):
        """
        Args:
            low (any): The lower bound (see Valid Inputs for more information).
            high (any): The upper bound (see Valid Inputs for more information).
            shape (tuple): The shape of this space.
            dtype (np.type): The data type (as numpy type) for this Space.
                Allowed are: np.int8,16,32,64, np.float16,32,64 and np.bool_.

        Valid inputs:
            BoxSpace(0.0, 1.0) # low and high are given as scalars and shape is assumed to be ()
                -> single scalar between low and high.
            BoxSpace(-1.0, 1.0, (3,4)) # low and high are scalars, and shape is provided -> nD array
                where all(!) elements are between low and high.
            BoxSpace(np.array([-1.0,-2.0]), np.array([2.0,4.0])) # low and high are arrays of the same shape
                (no shape given!) -> nD array where each dimension has different bounds.
        """
        super(BoxSpace, self).__init__(add_batch_rank=add_batch_rank, add_time_rank=add_time_rank,
                                       time_major=time_major)

        self.dtype = dtype

        # Determine the shape.
        if shape is None:
            if isinstance(low, (int, float, bool)):
                self._shape = ()
            else:
                self._shape = np.shape(low)
        else:
            assert isinstance(shape, tuple), "ERROR: `shape` must be None or a tuple."
            self._shape = shape

        # False if bounds are individualized (each dimension has its own lower and upper bounds and we can get
        # the single values from self.low and self.high), or a tuple of the globally valid low/high values that apply
        # to all values in all dimensions.
        self.global_bounds = None

        # Determine the bounds.
        # 0D Space.
        if self._shape == ():
            assert isinstance(low, (int, float, bool))
            self.global_bounds = (low, high)
        # nD Space (n > 0). Bounds can be single number or individual bounds.
        else:
            # Low/high values are given individually per item.
            if isinstance(low, (list, tuple, np.ndarray)):
                self.global_bounds = False
            # Only one low/high value. Use these as generic bounds for all values.
            else:
                assert np.isscalar(low) and np.isscalar(high)
                self.global_bounds = (low, high)

        self.low = np.array(low)
        self.high = np.array(high)
        assert self.low.shape == self.high.shape

    def force_batch(self, samples):
        assert self.has_time_rank is False, "ERROR: Cannot force a batch rank if Space `has_time_rank` is True!"
        # 0D (means: certainly no batch rank) or no extra rank given (compared to this Space), add a batch rank.
        if np.asarray(samples).ndim == 0 or \
                np.asarray(samples).ndim == len(self.get_shape(with_batch_rank=False, with_time_rank=False)):
            return np.array([samples])  # batch size=1
        # Samples is a list (whose len is interpreted as the batch size) -> return as np.array.
        elif isinstance(samples, list):
            return np.asarray(samples)
        return samples

    def get_shape(self, with_batch_rank=False, with_time_rank=False, time_major=None, **kwargs):
        if with_batch_rank is not False:
            # None shapes are typically only allowed in static graphs.
            if get_backend() == "tf":
                batch_rank = (((None,) if with_batch_rank is True else (with_batch_rank,))
                              if self.has_batch_rank else ())
            elif get_backend() == "pytorch":
                batch_rank = (((1,) if with_batch_rank is True else (with_batch_rank,))
                              if self.has_batch_rank else ())
        else:
            batch_rank = ()

        if with_time_rank is not False:
            time_rank = (((None,) if with_time_rank is True else (with_time_rank,))
                          if self.has_time_rank else ())
        else:
            time_rank = ()

        time_major = self.time_major if time_major is None else time_major
        if time_major is False:
            return batch_rank + time_rank + self.shape
        else:
            return time_rank + batch_rank + self.shape

    @property
    def flat_dim(self):
        return int(np.prod(self.shape))  # also works for shape=()

    @property
    def bounds(self):
        return self.low, self.high

    def get_variable(self, name, is_input_feed=False, add_batch_rank=None, add_time_rank=None,
                     time_major=None, is_python=False, local=False, **kwargs):
        add_batch_rank = self.has_batch_rank if add_batch_rank is None else add_batch_rank
        batch_rank = () if add_batch_rank is False else (None,) if add_batch_rank is True else (add_batch_rank,)

        add_time_rank = self.has_time_rank if add_time_rank is None else add_time_rank
        time_rank = () if add_time_rank is False else (None,) if add_time_rank is True else (add_time_rank,)

        time_major = self.time_major if time_major is None else time_major

        if time_major is False:
            shape = batch_rank + time_rank + self.shape
        else:
            shape = time_rank + batch_rank + self.shape

        if is_python is True or get_backend() == "python":
            if isinstance(add_batch_rank, int):
                if isinstance(add_time_rank, int):
                    if time_major:
                        var = [[0 for _ in range_(add_batch_rank)] for _ in range_(add_time_rank)]
                    else:
                        var = [[0 for _ in range_(add_time_rank)] for _ in range_(add_batch_rank)]
                else:
                    var = [0 for _ in range_(add_batch_rank)]
            elif isinstance(add_time_rank, int):
                var = [0 for _ in range_(add_time_rank)]
            else:
                var = []

            # Un-indent and just directly construct pytorch?
            if get_backend() == "pytorch" and is_input_feed:
                # Convert to PyTorch tensor because PyTorch cannot use
                return torch.zeros(shape)
            else:
                # TODO also convert?
                return var

        elif get_backend() == "tf":
            import tensorflow as tf
            # TODO: re-evaluate the cutting of a leading '/_?' (tf doesn't like it)
            name = re.sub(r'^/_?', "", name)
            if is_input_feed:
                return tf.placeholder(dtype=dtype(self.dtype), shape=shape, name=name)
            else:
                init_spec = kwargs.pop("initializer", None)
                # Bools should be initializable via 0 or not 0.
                if self.dtype == np.bool_ and isinstance(init_spec, (int, float)):
                    init_spec = (init_spec != 0)

                if self.dtype == np.str_ and init_spec == 0:
                    initializer = None
                else:
                    initializer = Initializer.from_spec(shape=shape, specification=init_spec).initializer

                return tf.get_variable(
                    name, shape=shape, dtype=dtype(self.dtype), initializer=initializer,
                    collections=[tf.GraphKeys.GLOBAL_VARIABLES if local is False else tf.GraphKeys.LOCAL_VARIABLES],
                    **kwargs
                )

    def zeros(self, size=None):
        return self.sample(size=size, fill_value=0)

    def contains(self, sample):
        sample_shape = sample.shape if not isinstance(sample, int) else ()
        if sample_shape != self.shape:
            return False
        return (sample >= self.low).all() and (sample <= self.high).all()

    def __repr__(self):
        return "{}({} {} {}{})".format(
            type(self).__name__.title(), self.shape, str(self.dtype), "; +batch" if self.has_batch_rank else
            "", "; +time" if self.has_time_rank else ""
        )

    def __eq__(self, other):
        return isinstance(other, self.__class__) and \
               self.shape == other.shape and self.dtype == other.dtype
               # np.allclose(self.low, other.low) and np.allclose(self.high, other.high) and \

    def __hash__(self):
        if self.shape == ():
            return hash((self.low, self.high))
        return hash((tuple(self.low), tuple(self.high)))
