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
import unittest

from rlgraph.components import Component, Synchronizable
from rlgraph.spaces import FloatBox
from rlgraph.tests import ComponentTest
from rlgraph.utils.decorators import rlgraph_api

VARIABLE_NAMES = ["variable_to_sync1", "variable_to_sync2"]


class MyCompWithVars(Component):
    """
    The Component with variables to test. Synchronizable can be added later as a drop-in via add_component.
    """
    def __init__(self, initializer1=0.0, initializer2=1.0, synchronizable=False, **kwargs):
        super(MyCompWithVars, self).__init__(**kwargs)
        self.space = FloatBox(shape=(4, 5))
        self.initializer1 = initializer1
        self.initializer2 = initializer2
        self.dummy_var_1 = None
        self.dummy_var_2 = None

        if synchronizable is True:
            self.add_components(Synchronizable(), expose_apis="sync")

    def create_variables(self, input_spaces, action_space=None):
        # create some dummy var to sync from/to.
        self.dummy_var_1 = self.get_variable(name=VARIABLE_NAMES[0], from_space=self.space,
                                             initializer=self.initializer1, trainable=True)
        self.dummy_var_2 = self.get_variable(name=VARIABLE_NAMES[1], from_space=self.space,
                                             initializer=self.initializer2, trainable=True)


class TestSynchronizableComponent(unittest.TestCase):

    def test_values_api_method(self):
        # Proof that all Components can push out their variable values.
        component_to_test = MyCompWithVars(synchronizable=False)
        test = ComponentTest(component=component_to_test)

        # Test pulling the variable values from the sync_out socket.
        expected1 = np.zeros(shape=component_to_test.space.shape)
        expected2 = np.ones(shape=component_to_test.space.shape)
        expected = dict({"variable-to-sync1": expected1, "variable-to-sync2": expected2})

        test.test("_variables", expected_outputs=expected)

    def test_sync_functionality(self):
        # Two Components, one with Synchronizable dropped in:
        # A: Can only push out values.
        # B: To be synced by A's values.
        sync_from = MyCompWithVars(scope="sync-from")
        sync_to = MyCompWithVars(initializer1=8.0, initializer2=7.0, scope="sync-to", synchronizable=True)

        # Create a dummy test component that contains our two Synchronizables.
        container = Component(name="container")
        container.add_components(sync_from, sync_to)

        @rlgraph_api(component=container)
        def execute_sync(self):
            values_ = sync_from._variables()
            return sync_to.sync(values_)

        test = ComponentTest(component=container)

        # Test syncing the variable from->to and check them before and after the sync.
        # Before the sync.
        test.variable_test(sync_to.get_variables(VARIABLE_NAMES), {
            "sync-to/"+VARIABLE_NAMES[0]: np.full(shape=sync_from.space.shape, fill_value=8.0),
            "sync-to/"+VARIABLE_NAMES[1]: np.full(shape=sync_from.space.shape, fill_value=7.0)
        })

        # Now sync and re-check.
        test.test("execute_sync", expected_outputs=None)

        # After the sync.
        test.variable_test(sync_to.get_variables(VARIABLE_NAMES), {
            "sync-to/"+VARIABLE_NAMES[0]: np.zeros(shape=sync_from.space.shape),
            "sync-to/"+VARIABLE_NAMES[1]: np.ones(shape=sync_from.space.shape)
        })

    def test_sync_socket_between_2_identical_comps_that_have_vars_only_in_their_sub_comps(self):
        """
        Similar to the Policy scenario, where the Policy Component owns a NeuralNetwork (which has vars)
        and has to be synced with other Policies.
        """
        # Create 2x: A custom Component (with vars) that holds another Component (with vars).
        # Then sync between them.
        comp1 = MyCompWithVars(scope="A")
        comp1.add_components(MyCompWithVars(scope="sub-of-A-with-vars"))

        comp2_writable = MyCompWithVars(scope="B", initializer1=3.0, initializer2=4.2, synchronizable=True)
        comp2_writable.add_components(MyCompWithVars(scope="sub-of-B-with-vars", initializer1=5.0, initializer2=6.2))

        container = Component(comp1, comp2_writable, scope="container")

        @rlgraph_api(component=container)
        def execute_sync(self):
            values_ = comp1._variables()
            return comp2_writable.sync(values_)

        test = ComponentTest(component=container)

        # Before the sync.
        test.variable_test(comp2_writable.get_variables([
            "container/B/variable_to_sync1",
            "container/B/variable_to_sync2",
            "container/B/sub-of-B-with-vars/variable_to_sync1",
            "container/B/sub-of-B-with-vars/variable_to_sync2"
        ]), {
            "container/B/variable_to_sync1": np.full(shape=comp1.space.shape, fill_value=3.0, dtype=np.float32),
            "container/B/variable_to_sync2": np.full(shape=comp1.space.shape, fill_value=4.2, dtype=np.float32),
            "container/B/sub-of-B-with-vars/variable_to_sync1": np.full(shape=comp1.space.shape, fill_value=5.0,
                                                                        dtype=np.float32),
            "container/B/sub-of-B-with-vars/variable_to_sync2": np.full(shape=comp1.space.shape, fill_value=6.2,
                                                                        dtype=np.float32)
        })

        # Now sync and re-check.
        test.test(("execute_sync", None), expected_outputs=None)

        # After the sync.
        test.variable_test(comp2_writable.get_variables([
            "container/B/variable_to_sync1",
            "container/B/variable_to_sync2",
            "container/B/sub-of-B-with-vars/variable_to_sync1",
            "container/B/sub-of-B-with-vars/variable_to_sync2"
        ]), {
            "container/B/variable_to_sync1": np.zeros(shape=comp1.space.shape, dtype=np.float32),
            "container/B/variable_to_sync2": np.ones(shape=comp1.space.shape, dtype=np.float32),
            "container/B/sub-of-B-with-vars/variable_to_sync1": np.zeros(shape=comp1.space.shape, dtype=np.float32),
            "container/B/sub-of-B-with-vars/variable_to_sync2": np.ones(shape=comp1.space.shape, dtype=np.float32)
        })




