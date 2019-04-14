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

from rlgraph.spaces.int_box import IntBox


class MultiDiscrete(IntBox):
    def __init__(self, high, shape=None, **kwargs):
        shape = shape or np.shape(high)
        assert len(shape) < 2, \
            "ERROR: `shape` for MultiDiscrete Space must be None, (), or (n,)!"
        super(MultiDiscrete, self).__init__(low=np.zeros_like(high), high=high, shape=shape, **kwargs)

    @property
    def flat_dim_with_categories(self):
        return int(np.sum(self.high))
