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

from yarl.spaces import *
from yarl.components import SplitterComponent
from .component_test import ComponentTest


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
            e=Discrete(2),
            f=Continuous(shape=(3,2)),
            g=Tuple(bool, Continuous(shape=())),
            add_batch_rank=True
        )

        # The Component to test.
        component_to_test = SplitterComponent(input_space=space)

        test = ComponentTest(component=component_to_test, input_spaces=dict(input=space))

        # Run the test.
        input_ = space.sample(size=3)

        test.test(out_socket_name="/a/ab", inputs=input_, expected_outputs=np.array(input_["a"]["ab"]))
        test.test(out_socket_name="/g/_T0_", inputs=input_, expected_outputs=np.array(input_["g"][0]))
        test.test(out_socket_name="/d", inputs=input_, expected_outputs=np.array(input_["d"]))
        test.test(out_socket_name="/f", inputs=input_, expected_outputs=np.array(input_["f"]))


