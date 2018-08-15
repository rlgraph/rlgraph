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

from rlgraph.utils.util import dtype as dtype_
from rlgraph.spaces.box_space import BoxSpace


class FloatBox(BoxSpace):
    def __init__(self, low=None, high=None, shape=None, dtype="float32", **kwargs):
        if low is None:
            assert high is None, "ERROR: If `low` is None, `high` must be None as well!"
            low = float("-inf")
            high = float("inf")
            self.unbounded = True
        else:
            self.unbounded = False
            # support calls like (FloatBox(1.0) -> low=0.0, high=1.0)
            if high is None:
                high = low
                low = 0.0

        dtype = dtype_(dtype, "np")
        assert dtype in [np.float16, np.float32, np.float64], "ERROR: FloatBox does not allow dtype '{}'!".format(dtype)

        super(FloatBox, self).__init__(low=low, high=high, shape=shape, dtype=dtype, **kwargs)

    def sample(self, size=None, fill_value=None):
        shape = self._get_np_shape(num_samples=size)
        if fill_value is not None:
            sample_ = np.full(shape=shape, fill_value=fill_value)
        else:
            if self.unbounded:
                sample_ = np.random.uniform(size=shape)
            else:
                sample_ = np.random.uniform(low=self.low, high=self.high, size=shape)

        # Make sure return values have the right dtype (float64 is np.random's default).
        return np.asarray(sample_, dtype=self.dtype)
