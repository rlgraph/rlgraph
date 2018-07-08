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


class MemSegmentTree(object):
    """
    In-memory Segment tree for prioritized replay.

    """

    def __init__(
            self,
            storage_variable,
            capacity=1048
    ):
        """
        Helper to represent a segment tree.

        Args:
            storage_variable: Memory content.
            capacity (int): Capacity of the segment tree.
        """
        self.values = storage_variable
        self.capacity = capacity

    def insert(self, index, element, insert_op=tf.add):
        """
        Inserts an element into the segment tree by determining
        its position in the tree.

        Args:
            index (int): Insertion index.
            element (any): Element to insert.
            insert_op (Union(tf.add, tf.minimum, tf, maximum)): Insert operation on the tree.
        """
        index += self.capacity
        # index = tf.Print(index, [index, self.values], summarize=1000, message='start index, values=')
        pass

    def get(self, index):
        """
        Reads an item from the segment tree.

        Args:
            index (int):

        Returns: The element.

        """
        return self.values[self.capacity + index]

    def index_of_prefixsum(self, prefix_sum):
        """
        Identifies the highest index which satisfies the condition that the sum
        over all elements from 0 till the index is <= prefix_sum.

        Args:
            prefix_sum .float): Upper bound on prefix we are allowed to select.

        Returns:
            int: Index/indices satisfying prefix sum condition.
        """
        pass

    def reduce(self, start, limit, reduce_op=tf.add):
        """
        Applies an operation to specified segment.

        Args:
            start (int): Start index to apply reduction to.
            limit (end): End index to apply reduction to.
            reduce_op (Union(tf.add, tf.minimum, tf.maximum)): Reduce op to apply.

        Returns:
            Number: Result of reduce operation
        """
        # Init result with neutral element of reduce op.
        # Note that all of these are commutative reduce ops.
        pass

    def get_min_value(self):
        """
        Returns min value of storage variable.
        """
        pass

    def get_sum(self):
        """
        Returns min value of storage variable.
        """
        pass
