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

from rlgraph.components.layers.preprocessing.container_splitter import ContainerSplitter
from rlgraph.spaces import *
from rlgraph.tests import ComponentTest


class TestContainerSplitterComponents(unittest.TestCase):
    """
    Tests the ContainerSplitter Component.
    """

    def test_dict_splitter(self):
        space = Dict(
            a=dict(aa=bool, ab=float),
            b=dict(ba=bool),
            c=float,
            d=IntBox(low=0, high=255),
            e=IntBox(2),
            f=FloatBox(shape=(3, 2)),
            g=Tuple(bool, FloatBox(shape=())),
            add_batch_rank=True
        )
        # Define the output-order.
        splitter = ContainerSplitter("g", "a", "b", "c", "d", "e", "f")
        test = ComponentTest(component=splitter, input_spaces=dict(inputs=space))

        # Get a batch of samples.
        input_ = space.sample(size=3)
        expected_output = [
            input_["g"],
            input_["a"],
            input_["b"],
            input_["c"],
            input_["d"],
            input_["e"],
            input_["f"]
        ]
        test.test(("call", input_), expected_outputs=expected_output)

    def test_dict_splitter_with_different_input_space(self):
        space = Dict(
            a=Tuple(bool, FloatBox(shape=())),
            b=FloatBox(shape=()),
            c=bool,
            d=IntBox(low=0, high=255),
            e=dict(ea=float),
            f=FloatBox(shape=(3, 2)),
            add_batch_rank=False
        )
        # Define the output-order.
        splitter = ContainerSplitter("b", "c", "d", "a", "f", "e")
        test = ComponentTest(component=splitter, input_spaces=dict(inputs=space))

        # Single sample (no batch rank).
        input_ = space.sample()
        expected_outputs = [
            input_["b"],
            input_["c"],
            input_["d"],
            input_["a"],
            input_["f"],
            input_["e"]
        ]

        test.test(("call", input_), expected_outputs=expected_outputs)

    def test_tuple_splitter(self):
        space = Tuple(FloatBox(shape=()), bool, IntBox(low=0, high=255), add_batch_rank=True)
        # Define the output-order.
        splitter = ContainerSplitter(tuple_length=len(space))
        test = ComponentTest(component=splitter, input_spaces=dict(inputs=space))

        # Single sample (batch size=6).
        input_ = space.sample(size=6)
        expected_outputs = [
            input_[0],
            input_[1],
            input_[2]
        ]

        test.test(("call", (input_,)), expected_outputs=expected_outputs)
