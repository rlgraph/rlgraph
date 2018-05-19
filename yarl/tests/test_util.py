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
    record_sample['terminal'] = np.zeros(num_samples)

    return record_sample


def terminal_records(record_space, num_samples):
    """
    Samples a number of records and enforces all terminals to be 1,
    which is needed for testing memories.

    Args:
        record_space (Space): Space to sample from.
        num_samples (int): Number of samples to draw.

    Returns:
        Dict: Sampled records with all terminal values set to 1.
    """
    record_sample = record_space.sample(size=num_samples)
    record_sample['terminal'] = np.ones(num_samples)

    return record_sample