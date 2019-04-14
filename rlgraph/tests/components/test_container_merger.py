# Copyright 2018/2019 The RLgraph authors. All Rights Reserved.
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

import unittest

from rlgraph.components.common.container_merger import ContainerMerger
from rlgraph.spaces import *
from rlgraph.tests import ComponentTest


class TestContainerMergerComponents(unittest.TestCase):
    """
    Tests the ContainerMerger Component.
    """

    def test_dict_merger_component(self):
        space = Tuple(
            dict(a=bool, b=float),
            dict(c=bool),
            float,
            IntBox(low=0, high=255),
            IntBox(2),
            FloatBox(shape=(3, 2)),
            Dict(d=bool, e=FloatBox(shape=())),
            add_batch_rank=False
        )
        merger = ContainerMerger("1", "2", "3", "test", "5", "6", "7")
        test = ComponentTest(component=merger, input_spaces=dict(inputs=[s for s in space]))

        # Get a single sample.
        sample = space.sample()
        expected_outputs = dict({"1": sample[0],
                                 "2": sample[1],
                                 "3": sample[2],
                                 "test": sample[3],
                                 "5": sample[4],
                                 "6": sample[5],
                                 "7": sample[6]})

        test.test(("merge", list(sample)), expected_outputs=expected_outputs)

    def test_tuple_merger_component(self):
        space = Tuple(
            dict(a=bool, b=float),
            dict(c=bool),
            float,
            IntBox(low=0, high=255),
            IntBox(2),
            FloatBox(shape=(3, 2)),
            Dict(d=bool, e=FloatBox(shape=())),
            add_batch_rank=False
        )
        merger = ContainerMerger(7)
        test = ComponentTest(component=merger, input_spaces=dict(inputs=[s for s in space]))

        # Get a single sample.
        sample = space.sample()
        expected_outputs = tuple([sample[0], sample[1], sample[2], sample[3], sample[4], sample[5], sample[6]])

        test.test(("merge", list(sample)), expected_outputs=expected_outputs)

    def test_tuple_merger_component_merging_two_data_op_tuples(self):
        space = Tuple(
            Tuple(IntBox(2), IntBox(3)),
            IntBox(3),
            Tuple(FloatBox(shape=(3,)), BoolBox(shape=(1,))),
            add_batch_rank=False
        )
        merger = ContainerMerger(merge_tuples_into_one=True)
        test = ComponentTest(component=merger, input_spaces=dict(inputs=[s for s in space]))

        # Get a single sample.
        sample = space.sample()
        expected_outputs = tuple([sample[0][0], sample[0][1], sample[1], sample[2][0], sample[2][1]])

        test.test(("merge", list(sample)), expected_outputs=expected_outputs)
