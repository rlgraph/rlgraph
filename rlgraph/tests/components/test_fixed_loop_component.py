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

import unittest

from rlgraph.components import Sampler, FixedLoop, LinearDecay
from rlgraph.spaces import BoolBox, Dict, IntBox
from rlgraph.tests import ComponentTest


class TestFixedLoopComponent(unittest.TestCase):
    """
    Tests the fixed loop component.
    """
    def test_fixed_loop_component(self):
        pass
    #    time_step_space = IntBox(1000, add_batch_rank=False)
    #    call_component = LinearDecay(from_=1.0, to_=0.0, start_timestep=0, num_timesteps=10)

    #    # Calls decay 10 times with same parameters.
    #    loop = FixedLoop(
    #        num_iterations=10,
    #        call_component=call_component,
    #        graph_fn_name="decayed_value"
    #    )
    #    loop.connect((loop, "api_methods"), (call_component, "time_step"))
    #    test = ComponentTest(component=loop, input_spaces=dict(
    #        time_step=time_step_space,
    #        inputs=time_step_space
    #    ))

    #    sample = test.test(("fixed_loop_result", 0), expected_outputs=None)

    #    print(sample)
