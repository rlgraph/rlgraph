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

from yarl.components import Component
from yarl.components.common import Synchronizable
from yarl.spaces import FloatBox
from yarl.tests import ComponentTest

VARIABLE_NAMES = ["variable_to_synch1", "variable_to_synch2"]


class MySyncComp(Synchronizable):
    """
    The Component to test. Synchronizable is only a mix-in class.
    """
    def __init__(self, initializer1=0.0, initializer2=1.0, **kwargs):
        super(MySyncComp, self).__init__(**kwargs)
        self.space = FloatBox(shape=(4, 5))
        self.initializer1 = initializer1
        self.initializer2 = initializer2
        self.dummy_var_1 = None
        self.dummy_var_2 = None

    def create_variables(self, input_spaces):
        # create some dummy var to synch from/to.
        self.dummy_var_1 = self.get_variable(name=VARIABLE_NAMES[0], from_space=self.space,
                                             initializer=self.initializer1)
        self.dummy_var_2 = self.get_variable(name=VARIABLE_NAMES[1], from_space=self.space,
                                             initializer=self.initializer2)


class TestSynchronizableComponent(unittest.TestCase):

    def test_synch_out_socket(self):
        # A Synchronizable that can only push out values (cannot be synched from another Synchronizable).
        component_to_test = MySyncComp(writable=False)
        test = ComponentTest(component=component_to_test)

        # Test pulling the variable values from the synch_out socket.
        expected1 = np.zeros(shape=component_to_test.space.shape)
        expected2 = np.ones(shape=component_to_test.space.shape)
        expected = dict(variable_to_synch1=expected1, variable_to_synch2=expected2)
        test.test(out_socket_names="synch_out", inputs=None, expected_outputs=expected)

    def test_synch_socket(self):
        # Two Synchronizables, A that can only push out values, B to be synched by A's values.
        synch_from = MySyncComp(writable=False, scope="synch-from")
        synch_to = MySyncComp(initializer1=8.0, initializer2=7.0, writable=True, scope="synch-to")
        # Create a dummy test component that contains our two Synchronizables.
        component_to_test = Component(name="dummy-comp")
        component_to_test.define_outputs("do_the_synch")
        component_to_test.add_components(synch_from, synch_to)
        # connect everything correctly
        component_to_test.connect((synch_from, "synch_out"), (synch_to, "synch_in"))
        component_to_test.connect((synch_to, "synch"), "do_the_synch")
        test = ComponentTest(component=component_to_test)

        # Test synching the variable from->to and check them before and after the synch.
        variables_dict = synch_to.get_variables(VARIABLE_NAMES)
        var1_value, var2_value = test.get_variable_values(*list(variables_dict.values()))

        expected1 = np.full(shape=synch_from.space.shape, fill_value=8.0)
        expected2 = np.full(shape=synch_from.space.shape, fill_value=7.0)
        test.assert_equal(var1_value, expected1)
        test.assert_equal(var2_value, expected2)

        # Now synch and re-check.
        test.test(out_socket_names="do_the_synch", inputs=None, expected_outputs=None)

        var1_value, var2_value = test.get_variable_values(*list(variables_dict.values()))

        expected1 = np.zeros(shape=synch_from.space.shape)
        expected2 = np.ones(shape=synch_from.space.shape)
        test.assert_equal(var1_value, expected1)
        test.assert_equal(var2_value, expected2)
