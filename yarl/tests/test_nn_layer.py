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

from yarl.components.layers import DenseLayer
from yarl.spaces import Dict, Tuple, Continuous
from yarl.tests import ComponentTest

import numpy as np


space = Continuous(shape=(1,2))  #Dict(a=Continuous(), b=Continuous(), c=Dict(d=Continuous()))

# The Component to test.
# - fixed 1.0 weights, no biases
component_to_test = DenseLayer(input_space=space, units=2, weights_spec=1.0, biases_spec=False)

# TODO: discuss, whether it would be better in the DenseLayer to wait until we know the input space.
# TODO: Maybe introduce a `at_build` method for components, in which they can do stuff after they know about the input Space.
# TODO: Then we wouldn't have to specify it twice here (once when we build the layer and once when we
# TODO: connect the core's input to the Space)!

# A ComponentTest object.
test = ComponentTest(component=component_to_test, input_spaces=dict(input=space))

# Run the test.
input_ = np.array([[0.5, 2.0]])
expected = np.array([[2.5, 2.5]])

result = test.test(out_socket_name="output", inputs=input_, expected_outputs=expected)
if result:
    print("test passed!")
else:
    print("test failed!")

