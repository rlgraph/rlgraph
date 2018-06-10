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

from yarl.components import Component, CONNECT_ALL, Synchronizable
from yarl.spaces import FloatBox
from yarl.tests import ComponentTest

VARIABLE_NAMES = ["variable_to_sync1", "variable_to_sync2"]


class MySyncComp(Component):
    """
    The Component to test. Synchronizable is only a mix-in class.
    """
    def __init__(self, initializer1=0.0, initializer2=1.0, synchronizable=False, **kwargs):
        super(MySyncComp, self).__init__(**kwargs)
        self.space = FloatBox(shape=(4, 5))
        self.initializer1 = initializer1
        self.initializer2 = initializer2
        self.dummy_var_1 = None
        self.dummy_var_2 = None
        if synchronizable is True:
            self.add_component(Synchronizable(), connections=CONNECT_ALL)

    def create_variables(self, input_spaces, action_space):
        # create some dummy var to sync from/to.
        self.dummy_var_1 = self.get_variable(name=VARIABLE_NAMES[0], from_space=self.space,
                                             initializer=self.initializer1)
        self.dummy_var_2 = self.get_variable(name=VARIABLE_NAMES[1], from_space=self.space,
                                             initializer=self.initializer2)


class TestSynchronizableComponent(unittest.TestCase):

    def test_sync_out_socket(self):
        # A Synchronizable that can only push out values (cannot be synced from another Synchronizable).
        component_to_test = MySyncComp(synchronizable=False)
        test = ComponentTest(component=component_to_test)

        # Test pulling the variable values from the sync_out socket.
        expected1 = np.zeros(shape=component_to_test.space.shape)
        expected2 = np.ones(shape=component_to_test.space.shape)
        expected = dict(variable_to_sync1=expected1, variable_to_sync2=expected2)
        test.test(out_socket_names="_variables", inputs=None, expected_outputs=expected)

    def test_sync_socket(self):
        # Two Synchronizables, A that can only push out values, B to be synced by A's values.
        sync_from = MySyncComp(scope="sync-from")
        sync_to = MySyncComp(initializer1=8.0, initializer2=7.0, scope="sync-to")
        # Add the Synchronizable to sync_to.
        sync_to.add_component(Synchronizable(), connections=CONNECT_ALL)
        # Create a dummy test component that contains our two Synchronizables.
        component_to_test = Component(name="dummy-comp")
        component_to_test.define_outputs("do_the_sync")
        component_to_test.add_components(sync_from, sync_to)
        # connect everything correctly
        component_to_test.connect((sync_from, "_variables"), (sync_to, "_values"))
        component_to_test.connect((sync_to, "sync"), "do_the_sync")
        test = ComponentTest(component=component_to_test)

        # Test syncing the variable from->to and check them before and after the sync.
        variables_dict = sync_to.get_variables(VARIABLE_NAMES)
        var1_value, var2_value = test.get_variable_values(*list(variables_dict.values()))

        expected1 = np.full(shape=sync_from.space.shape, fill_value=8.0)
        expected2 = np.full(shape=sync_from.space.shape, fill_value=7.0)
        test.assert_equal(var1_value, expected1)
        test.assert_equal(var2_value, expected2)

        # Now sync and re-check.
        test.test(out_socket_names="do_the_sync", inputs=None, expected_outputs=None)

        var1_value, var2_value = test.get_variable_values(*list(variables_dict.values()))

        expected1 = np.zeros(shape=sync_from.space.shape)
        expected2 = np.ones(shape=sync_from.space.shape)
        test.assert_equal(var1_value, expected1)
        test.assert_equal(var2_value, expected2)
