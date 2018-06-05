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

import logging
import numpy as np
import unittest

from yarl.components import Component, CONNECT_INS, CONNECT_OUTS
from yarl.tests import ComponentTest, root_logger

from .dummy_components import Dummy1to1, Dummy2to1, Dummy1to2


class TestSocketGraphFnConnections(unittest.TestCase):
    """
    Tests for different ways to connect Sockets to other Sockets (of other Components) and GraphFunctions.
    """
    root_logger.setLevel(level=logging.INFO)

    def test_1to1_component(self):
        """
        Adds a single component with 1-to-1 graph_fn to the core and passes a value through it.
        """
        component = Dummy1to1(scope="dummy")
        test = ComponentTest(component=component, input_spaces=dict(input=float))

        # expected output: input + 1.0 + 1.0
        test.test(out_socket_names="output", inputs=1.0, expected_outputs=2.0)
        test.test(out_socket_names="output", inputs=-5.0, expected_outputs=-4.0)

    def test_2to1_component(self):
        """
        Adds a single component with 2-to-1 graph_fn to the core and passes 2 values through it.
        """
        component = Dummy2to1(scope="dummy")
        test = ComponentTest(component=component, input_spaces=dict(input1=float, input2=float))

        # expected output: input1 + input2
        test.test(out_socket_names="output", inputs=dict(input1=1.0, input2=2.9), expected_outputs=3.9)
        test.test(out_socket_names="output", inputs=dict(input1=4.9, input2=-0.1),
                  expected_outputs=np.array(4.8, dtype=np.float32))

    def test_1to2_component(self):
        """
        Adds a single component with 1-to-2 graph_fn to the core and passes a value through it.
        """
        component = Dummy1to2(scope="dummy")
        test = ComponentTest(component=component, input_spaces=dict(input=float))

        # expected outputs: (input, input+1.0)
        test.test(out_socket_names=["output1", "output2"], inputs=1.0, expected_outputs=[1.0, 2.0])
        test.test(out_socket_names=["output1", "output2"], inputs=4.5, expected_outputs=[4.5, 5.5])

    def test_connecting_two_1to1_components(self):
        """
        Adds two components with 1-to-1 graph_fns to the core, connects them and passes a value through it.
        """
        core = Component()
        sub_comp1 = Dummy1to1(scope="comp1")
        sub_comp2 = Dummy1to1(scope="comp2")
        core.add_component(sub_comp1, connections=CONNECT_INS)
        core.add_component(sub_comp2, connections=CONNECT_OUTS)
        core.connect(sub_comp1, sub_comp2)

        test = ComponentTest(component=core, input_spaces=dict(input=float))

        # expected output: input + 1.0 + 1.0
        test.test(out_socket_names="output", inputs=1.0, expected_outputs=3.0)
        test.test(out_socket_names="output", inputs=-5.0, expected_outputs=-3.0)

