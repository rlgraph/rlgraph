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

import tensorflow as tf
import numpy as np

from yarl import Specifiable
from yarl.components.memories.segment_tree import SegmentTree
from yarl.utils.ops import FlattenedDataOp
from yarl.utils.util import get_batch_size


class MemPrioritizedReplay(Specifiable):
    """
    Implements an in-memory  prioritized replay.

    API:
        update_records(indices, update) -> Updates the given indices with the given priority scores.
    """
    def __init__(self, capacity=1000, next_states=True, alpha=1.0, beta=0.0):
        pass

    def insert_records(self, records):
        pass

    def get_records(self, num_records):
        pass

    def update_records(self, indices, update):
        pass
