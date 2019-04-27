# Copyright 2018/2019 The RLgraph authors. All Rights Reserved.
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

from rlgraph import get_backend
from rlgraph.spaces.box_space import BoxSpace
from rlgraph.spaces.float_box import FloatBox
from rlgraph.utils.util import convert_dtype as dtype_, LARGE_INTEGER


class IntBox(BoxSpace):
    """
    A box in Z^n (only integers; each coordinate is bounded)
    e.g. an image (w x h x RGB) where each color channel pixel can be between 0 and 255.
    """
    def __init__(self, low=None, high=None, shape=None, dtype="int32", **kwargs):
        """
        Valid inputs:
            IntBox(6)  # only high is given -> low assumed to be 0 (0D scalar).
            IntBox(0, 2) # low and high are given as scalars and shape is assumed to be 0D scalar.
            IntBox(-1, 1, (3,4)) # low and high are scalars, and shape is provided.
            IntBox(np.array([-1,-2]), np.array([2,4])) # low and high are arrays of the same shape (no shape given!)

        NOTE: The `high` value for IntBoxes is excluded. Valid values thus are from the interval: [low,high[
        """
        if low is None:
            if high is not None:
                low = 0
            else:
                low = -LARGE_INTEGER
                high = LARGE_INTEGER
        # support calls like (IntBox(5) -> low=0, high=5)
        elif high is None:
            high = low
            low = 0

        dtype = dtype_(dtype, "np")
        assert dtype in [np.int16, np.int32, np.int64, np.uint8], \
            "ERROR: IntBox does not allow dtype '{}'!".format(dtype)

        super(IntBox, self).__init__(low=low, high=high, shape=shape, dtype=dtype, **kwargs)

        self.num_categories = None if self.global_bounds is False else self.global_bounds[1]

    def get_shape(self, with_batch_rank=False, with_time_rank=False, **kwargs):
        """
        Keyword Args:
            with_category_rank (bool): Whether to include a category rank for this IntBox (if all dims have equal
                lower/upper bounds).
        """
        with_category_rank = kwargs.pop("with_category_rank", False)
        shape = super(IntBox, self).get_shape(with_batch_rank=with_batch_rank, with_time_rank=with_time_rank, **kwargs)
        if with_category_rank is not False:
            return shape + ((self.num_categories,) if self.num_categories is not None else ())
        return shape

    def as_one_hot_float_space(self):
        """
        Returns a new FloatBox Space resulting from one-hot flattening out this space
        along its number of categories.

        Returns:
            FloatBox: The resulting FloatBox Space (with the same batch and time-rank settings).
        """
        return FloatBox(
            low=0.0, high=1.0, shape=self.get_shape(with_category_rank=True),
            add_batch_rank=self.has_batch_rank, add_time_rank=self.has_time_rank
        )

    @property
    def flat_dim_with_categories(self):
        """
        If we were to flatten this Space and also consider each single possible int value (assuming global bounds)
        as one category, what would the dimension have to be to represent this Space?
        """
        if self.global_bounds is False:
            return int(np.sum(self.high))  # TODO: this assumes that low is always 0.
        return int(np.prod(self.shape) * self.global_bounds[1])

    def sample(self, size=None, fill_value=None):
        shape = self._get_np_shape(num_samples=size)
        if fill_value is None:
            sample_ = np.random.uniform(low=self.low, high=self.high, size=shape)
        else:
            sample_ = fill_value if shape == () or shape is None else np.full(shape=shape, fill_value=fill_value)

        return np.asarray(sample_, dtype=self.dtype)

    def get_variable(
            self, name, is_input_feed=False, add_batch_rank=None, add_time_rank=None, time_major=None, is_python=False,
            local=False, **kwargs
    ):
        variable = super(IntBox, self).get_variable(
            name, is_input_feed, add_batch_rank, add_time_rank, time_major, is_python, local, **kwargs
        )
        # Add num_categories information to placeholder.
        if get_backend() == "tf" and is_python is False:
            variable._num_categories = self.num_categories
        return variable

    def contains(self, sample):
        # If int: Check for int type in given sample.
        if not np.equal(np.mod(sample, 1), 0).all():
            return False
        return super(IntBox, self).contains(sample)

