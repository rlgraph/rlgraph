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

    TODO check how to manage as component.
    """
    def __init__(
        self,
        storage_variable,
        capacity=1048
    ):
        """
        Helper to represent a segment tree in pure TensorFlow.

        Args:
            storage_variable (tf.Variable): TensorFlow variable to use for storage.
            capacity (int): Capacity of the segment tree.
        """
        self.values = storage_variable
        self.capacity = capacity

    # TODO note we do not want to override _setitem_ because of TF
    # variable/tensor semantics -> variables cannot be assigned without tf.assign
    def insert(self, index, element, insert_op=tf.add):
        """
        Inserts an element into the segment tree by determining
        its position in the tree.

        Args:
            index (int): Insertion index.
            element (any): Element to insert.
            op Union(tf.add, tf.min): Insert operation on the tree.
        """
        index += self.capacity
        assignments = list()

        # TODO replace with component assign utility.
        assignments.append(tf.assign(ref=self.values, value=element))

        # Search and update values while index >=1
        loop_update_index = index / 2

        def insert_body(loop_update_index, assignments):
            with tf.control_dependencies(control_inputs=assignments):
                update_val = insert_op(
                    x=self.values[2 * loop_update_index],
                    y=self.values[2 * loop_update_index]
                )
                assignments.append(tf.assign(ref=self.values, value=update_val))

            return loop_update_index / 2, assignments

        def cond(loop_update_index, assignments):
            return loop_update_index >= 1

        with tf.control_dependencies(control_inputs=assignments):
            _, assignments = tf.while_loop(cond=cond, body=insert_body, loop_vars=(loop_update_index, list()))

        return assignments

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

        Args:
            prefix_sum:

        Returns:

        """
        pass

    def reduce_sum(self, start, limit):
        """
        Applies sum() to specified segment.

        Args:
            start (int): Start index to apply reduction to.
            limit (end): End index to apply reduction to.

        Returns:
            any: Result of reduce

        """
        pass

    def reduce_min(self, start, limit):
        """
        Applies min() to specified segment.

        Args:
            start (int): Start index to apply reduction to.
            limit (end): End index to apply reduction to.

        Returns:
            any: Result of reduce

        """
        pass
