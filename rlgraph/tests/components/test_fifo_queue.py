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
import threading
import time
import unittest

from rlgraph.components.memories.fifo_queue import FIFOQueue
from rlgraph.spaces import Dict, BoolBox, Tuple
from rlgraph.tests import ComponentTest
from rlgraph.utils.ops import flatten_op, unflatten_op


class TestFIFOQueue(unittest.TestCase):
    """
    Tests sampling and insertion behaviour of the FIFOQueue class.
    """
    record_space = Dict(
        states=dict(state1=float, state2=float, state3=bool),
        actions=dict(action1=float, action2=Tuple(float, float)),
        reward=float,
        terminals=BoolBox(),
        add_batch_rank=True
    )
    capacity = 10

    input_spaces = dict(
        records=record_space,
        num_records=int
    )

    def test_enqueue_dequeue(self):
        """
        Simply tests insert op without checking internal logic.
        """
        fifo_queue = FIFOQueue(capacity=self.capacity, record_space=self.record_space)
        test = ComponentTest(component=fifo_queue, input_spaces=self.input_spaces)

        first_record = self.record_space.sample(size=1)
        test.test(("insert_records", first_record), expected_outputs=None)
        test.test("get_size", expected_outputs=1)

        further_records = self.record_space.sample(size=5)
        test.test(("insert_records", further_records), expected_outputs=None)
        test.test("get_size", expected_outputs=6)

        expected = dict()
        for (k1, v1), (k2, v2) in zip(flatten_op(first_record).items(), flatten_op(further_records).items()):
            expected[k1] = np.concatenate((v1, v2[:4]))
        expected = unflatten_op(expected)

        test.test(("get_records", 5), expected_outputs=expected)
        test.test("get_size", expected_outputs=1)

    def test_capacity(self):
        """
        Tests if insert correctly blocks when capacity is reached.
        """
        fifo_queue = FIFOQueue(capacity=self.capacity, record_space=self.record_space)
        test = ComponentTest(component=fifo_queue, input_spaces=self.input_spaces)

        def run(expected_):
            # Wait n seconds.
            time.sleep(2)
            # Pull something out of the queue again to continue.
            test.test(("get_records", 2), expected_outputs=expected_)

        # Insert one more element than capacity
        records = self.record_space.sample(size=self.capacity + 1)

        expected = dict()
        for key, value in flatten_op(records).items():
            expected[key] = value[:2]
        expected = unflatten_op(expected)

        # Start thread to save this one from getting stuck due to capacity overflow.
        thread = threading.Thread(target=run, args=(expected,))
        thread.start()

        print("Going over capacity: blocking ...")
        test.test(("insert_records", records), expected_outputs=None)
        print("Dequeued some items in another thread. Unblocked.")

        thread.join()

    def test_fifo_queue_with_distributed_tf(self):
        """
        Tests if FIFO is correctly shared between two processes running in distributed tf.
        """
        cluster_spec = dict(source=["localhost:22222"], target=["localhost:22223"])

        def run1():
            fifo_queue_1 = FIFOQueue(capacity=self.capacity, device="/job:source/task:0/cpu")
            test_1 = ComponentTest(component=fifo_queue_1, input_spaces=self.input_spaces, execution_spec=dict(
                mode="distributed",
                distributed_spec=dict(job="source", task_index=0, cluster_spec=cluster_spec)
            ))
            # Insert elements from source.
            records = self.record_space.sample(size=self.capacity)
            print("inserting into source-side queue ...")
            test_1.test(("insert_records", records), expected_outputs=None)
            print("size of source-side queue:")
            print(test_1.test("get_size", expected_outputs=None))
            # Pull one sample out.
            print("pulling from source-side queue:")
            print(test_1.test(("get_records", 2), expected_outputs=None))

        def run2():
            fifo_queue_2 = FIFOQueue(capacity=self.capacity, device="/job:source/task:0/cpu")
            test_2 = ComponentTest(component=fifo_queue_2, input_spaces=self.input_spaces, execution_spec=dict(
                mode="distributed",
                distributed_spec=dict(job="target", task_index=0, cluster_spec=cluster_spec)
            ))
            # Dequeue elements in target.
            print("size of target-side queue:")
            print(test_2.test("get_size", expected_outputs=None))
            print("pulling from target-side queue:")
            print(test_2.test(("get_records", 5), expected_outputs=None))

        # Start thread to save this one from getting stuck due to capacity overflow.
        thread_1 = threading.Thread(target=run1)
        thread_2 = threading.Thread(target=run2)
        thread_1.start()
        thread_2.start()

        thread_1.join()
        thread_2.join()
