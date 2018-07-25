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


def non_terminal_records(record_space, num_samples):
    """
    Samples a number of records and enforces all terminals to be 0,
    which is needed for testing memories.

    Args:
        record_space (Space): Space to sample from.
        num_samples (int): Number of samples to draw.

    Returns:
        Dict: Sampled records with all terminal values set to 0.
    """
    record_sample = record_space.sample(size=num_samples)
    record_sample['terminals'] = np.full(shape=(num_samples,), fill_value=np.bool_(False))

    return record_sample


def terminal_records(record_space, num_samples):
    """
    Samples a number of records and enforces all terminals to be True,
    which is needed for testing memories.

    Args:
        record_space (Space): Space to sample from.
        num_samples (int): Number of samples to draw.

    Returns:
        Dict: Sampled records with all terminal values set to True.
    """
    record_sample = record_space.sample(size=num_samples)
    record_sample['terminals'] = np.full(shape=(num_samples,), fill_value=np.bool_(True))

    return record_sample


def recursive_assert_almost_equal(x, y, decimals=7):
    """
    Checks two structures (dict, DataOpDict, tuple, DataOpTuple, list, np.array, float, int, etc..) for (almost!)
    numeric identity.
    All numbers in the two structures have to match up to `decimal` digits after the floating point.
    Uses assertions (not boolean return).

    Args:
        x (any): The first value to be compared (to `y`).
        y (any): The second value to be compared (to `x`).
        decimals (int): The number of digits after the floating point up to which all numeric values have to match.
    """
    # A dict type.
    if isinstance(x, dict):
        assert isinstance(y, dict), "ERROR: If x is dict, y needs to be a dict as well!"
        for key, value in x.items():
            assert key in y, "ERROR: y does not have x's key='{}'!".format(key)
            recursive_assert_almost_equal(value, y[key], decimals=decimals)
    # A tuple type.
    elif isinstance(x, (tuple, list)):
        assert isinstance(y, (tuple, list)), "ERROR: If x is tuple, y needs to be a tuple as well!"
        assert len(y) == len(x), "ERROR: y does not have the same length as " \
                                 "x ({} vs {})!".format(len(y), len(x))
        for i, value in enumerate(x):
            recursive_assert_almost_equal(value, y[i], decimals=decimals)
    # Boolean stuff.
    elif isinstance(x, (np.bool_, bool)):
        assert bool(x) is bool(y), "ERROR: x ({}) is not y ({})!".format(x, y)
    # Everything else.
    else:
        np.testing.assert_almost_equal(x, y, decimal=decimals)
