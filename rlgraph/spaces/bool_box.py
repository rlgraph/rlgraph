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

from rlgraph.utils.util import dtype
from rlgraph.spaces.box_space import BoxSpace


class BoolBox(BoxSpace):
    def __init__(self, shape=None, add_batch_rank=False, add_time_rank=False):
        super(BoolBox, self).__init__(low=False, high=True, shape=shape, add_batch_rank=add_batch_rank,
                                      add_time_rank=add_time_rank, dtype="bool")

    def sample(self, size=None):
        shape = self._get_np_shape(num_samples=size)
        return np.random.choice(a=[False, True], size=shape)

    def contains(self, sample):
        if self.shape == ():
            return isinstance(sample, (bool, np.bool_))
        else:
            return dtype(sample.dtype, "np") == np.bool_

