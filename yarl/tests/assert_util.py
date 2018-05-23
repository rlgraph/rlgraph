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


def recursive_assert_almost_equal(x, y, decimal=7):
    """
    Checks two structures (dict, DataOpDict, tuple, list, np.array, float, int, etc..) for (almost!) numeric identity.
    All numbers in the two structures have to match up to `decimal` digits after the floating point.
    Uses assertions (not boolean return).

    Args:
        x (any): The first value to be compared (to `y`).
        y (any): The second value to be compared (to `x`).
        decimal (int): The number of digits after the floating point up to which all numeric values have to match.
    """
    # A dict type.
    if isinstance(x, dict):
        assert isinstance(y, dict), "ERROR: y needs to be a dict as well!"
        for k, v in x.items():
            assert k in y, "ERROR: y does not have x's key='{}'!".format(k)
            recursive_assert_almost_equal(v, y[k])
    # A tuple type.
    elif isinstance(x, tuple):
        assert isinstance(y, tuple), "ERROR: y needs to be a tuple as well!"
        assert len(y) == len(x), "ERROR: y does not have the same length as " \
                                 "x ({} vs {})!".format(len(y), len(x))
        for i, v in enumerate(x):
            recursive_assert_almost_equal(v, y[i])
    # Everything else.
    else:
        np.testing.assert_almost_equal(x, y, decimal=decimal)