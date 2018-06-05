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

import unittest

from yarl.components import Component, CONNECT_INS, CONNECT_OUTS
from yarl.tests import ComponentTest

from .dummy_components import Dummy1to1


class TestSocketGraphFnConnections(unittest.TestCase):
    """
    Tests for different ways to connect Sockets to other Sockets (of other Components) and GraphFunctions.
    """
    def test_connecting_two_components(self):
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

