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
import numpy as np

from yarl.components import Component
from yarl.components.layers.preprocessing import Flatten
from yarl.spaces import FloatBox
from yarl.tests import ComponentTest


class TestComponentCopy(unittest.TestCase):
    """
    Tests copying a constructed Component and adding the copy as well as the original into another Component.
    """
    def test_copying_a_component(self):
        # Flatten a simple 2x2 FloatBox to (4,).
        space = FloatBox(shape=(2,2), add_batch_rank=False)

        flatten_orig = Flatten()
        flatten_copy = flatten_orig.copy(scope="flatten-copy")
        component_to_test = Component(flatten_orig, flatten_copy,
                                      inputs=["input1", "input2"], outputs=["output1", "output2"],
                                      connections=[
                                          ["input1", ["flatten", "input"]],
                                          ["input2", ["flatten-copy", "input"]],
                                          [["flatten", "output"], "output1"],
                                          [["flatten-copy", "output"], "output2"]
                                      ])
        test = ComponentTest(component=component_to_test, input_spaces=dict(input1=space, input2=space))

        input_ = dict(
            input1=np.array([[0.5, 2.0], [1.0, 2.0]]),
            input2=np.array([[1.0, 2.0], [3.0, 4.0]])
        )
        expected = dict(
            output1=np.array([0.5, 2.0, 1.0, 2.0]),
            output2=np.array([1.0, 2.0, 3.0, 4.0])
        )
        for i in range(2):
            test.test(out_socket_name="output"+str(i+1), inputs=input_, expected_outputs=expected["output"+str(i+1)])

