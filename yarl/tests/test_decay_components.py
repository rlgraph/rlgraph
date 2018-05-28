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

from yarl.components.common.decay_components import *
from yarl.spaces import *

from .component_test import ComponentTest


class TestDecayComponents(unittest.TestCase):
    """
    Tests creation, sampling and shapes of Spaces.
    """
    def test_linear_decay(self):
        # Decaying a value always without batch dimension (does not make sense for global time step).
        time_step_space = IntBox(200, add_batch_rank=False)

        # The Component to test.
        decay_component = LinearDecay(from_=1.0, to_=0.0, start_timestep=100, num_timesteps=100)
        test = ComponentTest(component=decay_component, input_spaces=dict(time_step=time_step_space))

        # Values to pass as single items.
        input_ = np.array([0, 1, 2, 25, 50, 100, 110, 112, 120, 130, 150, 180, 190, 195, 200, 201, 210, 250, 1000])
        expected = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.88, 0.8, 0.7, 0.5, 0.2, 0.1, 0.05, 0.0, 0.0, 0.0,
                              0.0, 0.0])
        for i, e in zip(input_, expected):
            test.test(out_socket_name="value", inputs=i, expected_outputs=e)

