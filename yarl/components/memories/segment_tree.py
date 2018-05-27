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
            insert_op (Union(tf.add, tf.min)): Insert operation on the tree.
        """
        index += self.capacity
        assignments = list()

        # TODO replace with component assign utility.
        assignments.append(tf.scatter_update(ref=self.values, indices=index, updates=element))

        # Search and update values while index >=1
        loop_update_index = tf.div(x=index, y=2)

        def insert_body(loop_update_index, assignments):
            with tf.control_dependencies(control_inputs=assignments):
                update_val = insert_op(
                    x=self.values[2 * loop_update_index],
                    y=self.values[2 * loop_update_index + 1]
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
        Find the highest index which satisfies the condition that the sum
        over all elements from 0 till the index is <= prefix_sum.

        Args:
            prefix_sum (float): Upper bound on prefix we are allowed to select.

        Returns:
            int: Index satisfying prefix sum condition.
        """
        assert_ops = list()
        # 0 <= prefix_sum <= sum(priorities)
        assert_ops.append(tf.Assert(condition=tf.less_equal(x=prefix_sum, y=tf.reduce_sum(input_tensor=self.values))))
        assert_ops.append(tf.Assert(condition=tf.greater_equal(x=prefix_sum, y=0.0)))
        index = 1

        def search_body(index, prefix_sum):
            # Is the value at position 2 * index > prefix sum?
            compare_value = self.values[2 * index]

            def update_prefix_sum_fn(index, prefix_sum):
                # 'Use up' values in this segment, then jump to next.
                prefix_sum -= self.values[2 * index]
                return 2 * index + 1, prefix_sum

            index, prefix_sum = tf.cond(
                pred=compare_value > prefix_sum,
                # If over prefix sum, jump index.
                true_fn=lambda: (2 * index, prefix_sum),
                # Else adjust prefix sum until done.
                false_fn=update_prefix_sum_fn(index, prefix_sum)
            )
            return index, prefix_sum

        def cond(index, prefix_sum):
            return index < self.capacity

        with tf.control_dependencies(control_inputs=assert_ops):
            index, _ = tf.while_loop(cond=cond, body=search_body, loop_vars=[index, prefix_sum])

        return index - self.capacity

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
        if reduce_op == tf.add:
            result = 0
        elif reduce_op == tf.minimum:
            result = float('inf')
        elif reduce_op == tf.maximum:
            result = float('-inf')
        else:
            raise ValueError("Unsupported reduce OP. Support ops are [tf.add, tf.minimum, tf.maximum]")

        start += self.capacity
        limit += self.capacity

        def reduce_body(start, limit, result):
            start_mod = tf.mod(x=start, y=2)

            def update_start_fn(start, result):
                result = reduce_op(x=result, y=self.values[start])
                start += 1
                return start, result

            start, result = tf.cond(
                pred=start_mod == 0,
                true_fn=lambda: (start, result),
                false_fn=update_start_fn(start, result)
            )

            end_mod = tf.mod(x=limit, y=2)

            def update_limit_fn(limit, result):
                limit -= 1
                result = reduce_op(x=result, y=self.values[limit])

                return limit, result

            limit, result = tf.cond(
                pred=end_mod == 0,
                true_fn=lambda: (limit, result),
                false_fn=update_limit_fn(limit, result)
            )
            return start / 2, limit / 2, result

        def cond(start, limit, result):
            return start < limit

        _, _, result = tf.while_loop(cond=cond, body=reduce_body, loop_vars=(start, limit, result))

        return result
