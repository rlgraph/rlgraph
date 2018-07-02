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
import unittest

from yarl.components.common import Splitter, Merger
from yarl.spaces import *
from yarl.utils.ops import flatten_op
from yarl.tests import ComponentTest


class TestSplitterMergerComponents(unittest.TestCase):
    """
    Tests the Splitter- and Merger-Components.
    """

    def test_splitter_component(self):
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
        component_to_test = Splitter(input_space=space)
        test = ComponentTest(component=component_to_test, input_spaces=dict(split=space))

        # Get a batch of samples.
        input_ = space.sample(size=3)
        expected_output = [
            input_["a"]["aa"], input_["a"]["ab"],
            input_["b"]["ba"],
            input_["c"],
            input_["d"],
            input_["e"],
            input_["f"],
            input_["g"][0], input_["g"][1]
        ]
        test.test(api_method="split", params=input_, expected_outputs=expected_output)

    def test_splitter_component_with_custom_names(self):
        space = Dict(
            a=Tuple(bool, FloatBox(shape=())),
            b=FloatBox(shape=()),
            c=bool,
            d=IntBox(low=0, high=255),
            e=dict(ea=float),
            f=FloatBox(shape=(3, 2)),
            add_batch_rank=False
        )
        # Using custom output names.
        component_to_test = Splitter(input_space=space, output_names=["atup0bool", "atup1cont", "bcont",
                                                                      "cbool", "dintbox", "eeafloat",
                                                                      "fcont"])
        test = ComponentTest(component=component_to_test, input_spaces=dict(input=space))

        # Single sample (no batch rank).
        input_ = space.sample()

        # Also check the types each time.
        out = test.test(api_method="atup0bool", params=input_, expected_outputs=np.array(input_["a"][0]))
        self.assertTrue(isinstance(out.item(), bool))
        out = test.test(api_method="fcont", params=input_, expected_outputs=np.array(input_["f"]))
        self.assertTrue(isinstance(out, np.ndarray))
        out = test.test(api_method="eeafloat", params=input_, expected_outputs=np.array(input_["e"]["ea"]))
        self.assertTrue(isinstance(out.item(), float))
        out = test.test(api_method="cbool", params=input_, expected_outputs=np.array(input_["c"]))
        self.assertTrue(isinstance(out.item(), bool))

    def test_merger_component(self):
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
        component_to_test = Merger(output_space=space)
        flattened_space = space.flatten()
        test = ComponentTest(component=component_to_test, input_spaces=flattened_space)

        # Get a single sample.
        sample = space.sample()
        flattened_input = flatten_op(sample)

        test.test(out_socket_names="output", inputs=flattened_input, expected_outputs=sample)

    """
    FINISH THIS WHEN WE NEED IT. Doesn't seem to be clear on how to unflatten from custom names.
    def test_merger_component_with_custom_names(self):
        space = Tuple(
            Dict(a=bool, b=FloatBox(shape=(1,))),
            dict(c=bool, d=float),
            IntBox(low=0, high=255),
            dict(e=bool, f=bool),
            IntBox(6),
            add_batch_rank=True
        )
        component_to_test = Merger(output_space=space, input_names=["tup0a", "tup0b", "tup1c", "tup1d",
                                                                             "tup2", "tup3e", "tup3f", "tup4"])
        flattened_space = space.flatten()
        test = ComponentTest(component=component_to_test, input_spaces=flattened_space)

        # Get a batch of samples.
        sample = space.sample(size=2)
        flattened_input = flatten_op(sample)
        # Change the names from auto-generated to our manual ones.

        test.test(out_socket_names="output", api_methods=flattened_input, expected_outputs=sample)
    """
