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

from yarl.components.common import Synchronizable
from yarl.spaces import FloatBox

from .component_test import ComponentTest


class MySyncComp(Synchronizable):

    def __init__(self, **kwargs):
        super(MySyncComp, self).__init__(**kwargs)
        self.space = FloatBox(shape=(4, 5))
        self.dummy_var = None

    def create_variables(self, input_spaces):
        # create some dummy var to synch from/to.
        self.dummy_var = self.space.get_tensor_variable(name="variable_to_synch",
                                                        initializer=1.0)


class TestSynchronizableComponent(unittest.TestCase):

    def test_synch_out_socket(self):
        # A Synchronizable that can only push out values (cannot be synched from another Synchronizable).
        component_to_test = MySyncComp(writable=False)
        test = ComponentTest(component=component_to_test)

        # Test pulling the variable values from the synch_out socket.
        expected = np.ones(shape=(component_to_test.space.shape))
        test.test(out_socket_name="synch_out", inputs=None, expected_outputs=expected)

