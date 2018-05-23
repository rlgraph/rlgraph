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


class SegmentTree(object):
    """
    Segment tree for prioritized replay.
    """
    def __init__(
        self,
        capacity=1048
    ):
        self.capacity = capacity

    def insert(self, index, element):
        """
        Inserts an element into the segment tree by determining
        its position in the tree.

        Args:
            index (int): Insertion index.
            element (any): Element to insert.
        """
        pass

    def get(self, index):
        """
        Reads an item from the segment tree.

        Args:
            index (int):

        Returns: The element.

        """
        pass

    def reduce(self, start, limit):
        """
        #TODO figure out how to optimize this in tensorflow.

        Applies reduce op to specified segment.

        Args:
            start (int): Start index to apply reduction to.
            limit (end): End index to apply reduction to.

        Returns:
            any: Result of reduce

        """
        pass
