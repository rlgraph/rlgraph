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

import operator

from rlgraph.utils.rlgraph_errors import RLGraphError


class MemSegmentTree(object):
    """
    In-memory Segment tree for prioritized replay.

    Note: The pure TensorFlow segment tree is much slower because variable updating is expensive,
    and in scenarios like Ape-X, memory and update are separated processes, so there is little to be gained
    from inserting into the graph.
    """

    def __init__(
            self,
            values,
            capacity,
            operator=operator.add
    ):
        """
        Helper to represent a segment tree.

        Args:
            values (list): Storage for the segment tree.
            capacity (int): Capacity of segment tree.
            operator (callable): Reduce operation of the segment tree.
        """
        self.values = values
        self.capacity = capacity
        self.operator = operator

    def insert(self, index, element):
        """
        Inserts an element into the segment tree by determining
        its position in the tree.

        Args:
            index (int): Insertion index.
            element (any): Element to insert.
        """
        index += self.capacity
        self.values[index] = element

        #void modify(int p, int value) {  // set value at position p
        # for (t[p += n] = value; p > 1; p >>= 1) t[p>>1] = t[p] + t[p^1];
        # }

        # Bit shift should be slightly faster here than division.
        index = index >> 1
        while index >= 1:
            # No shift because small multiplications are optimized.
            update_index = 2 * index
            self.values[index] = self.operator(
                self.values[update_index],
                self.values[update_index + 1]
            )
            index = index >> 1

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
        assert 0 <= prefix_sum <= self.get_sum() + 1e-5
        index = 1

        while index < self.capacity:
            update_index = 2 * index
            if self.values[update_index] > prefix_sum:
                index = update_index
            else:
                prefix_sum -= self.values[update_index]
                index = update_index + 1
        return index - self.capacity

    def reduce(self, start, limit, reduce_op=operator.add):
        """
        Applies an operation to specified segment.

        Args:
            start (int): Start index to apply reduction to.
            limit (end): End index to apply reduction to.
            reduce_op (Union(operator.add, min, max)): Reduce op to apply.

        Returns:
            Number: Result of reduce operation
        """
        if limit is None:
            limit = self.capacity
        if limit < 0:
            limit += self.capacity

        # Init result with neutral element of reduce op.
        # Note that all of these are commutative reduce ops.
        if reduce_op == operator.add:
            result = 0.0
        elif reduce_op == min:
            result = float('inf')
        elif reduce_op == max:
            result = float('-inf')
        else:
            raise RLGraphError("Unsupported reduce OP. Support ops are [add, min, max].")

        start += self.capacity
        limit += self.capacity

        while start < limit:
            if start & 1:
                result = reduce_op(result, self.values[start])
                start += 1
            if limit & 1:
                limit -= 1
                result = reduce_op(result, self.values[limit])

            start = start >> 1
            limit = limit >> 1

        return result

    def get_min_value(self, start=0, stop=None):
        """
        Returns min value of storage variable.
        """
        return self.reduce(start, stop, reduce_op=min)

    def get_sum(self, start=0, stop=None):
        """
        Returns sum value of storage variable.
        """
        return self.reduce(start, stop, reduce_op=operator.add)


class MinSumSegmentTree(object):
    """
    This class merges two segment trees' operations for performance reasons to avoid
    unnecessary duplication of the insert loops.
    """

    def __init__(
            self,
            sum_tree,
            min_tree,
            capacity,
    ):
        self.sum_segment_tree = sum_tree
        self.min_segment_tree = min_tree
        self.capacity = capacity

    def insert(self, index, element):
        """
        Inserts an element into both segment trees by determining
        its position in the trees.

        Args:
            index (int): Insertion index.
            element (any): Element to insert.
        """
        index += self.capacity
        self.sum_segment_tree.values[index] = element
        self.min_segment_tree.values[index] = element

        index = index >> 1
        while index >= 1:
            update_index = 2 * index
            self.sum_segment_tree.values[index] = self.sum_segment_tree.values[update_index] +\
                self.sum_segment_tree.values[update_index + 1]
            self.min_segment_tree.values[index] = min(self.min_segment_tree.values[update_index],
                self.min_segment_tree.values[update_index + 1])
            index = index >> 1
