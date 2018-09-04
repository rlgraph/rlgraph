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

from rlgraph.components.common.staging_area import StagingArea
from rlgraph.spaces import *
from rlgraph.tests import ComponentTest, recursive_assert_almost_equal


class TestStagingArea(unittest.TestCase):

    def test_staging_area(self):
        input_spaces = Tuple(
            FloatBox(shape=(3, 2)),
            Dict(a=FloatBox(shape=(1,)), b=bool),
            bool,
            IntBox(shape=(2,))
        )
        staging_area = StagingArea(num_data=len(input_spaces))
        test = ComponentTest(component=staging_area, input_spaces=dict(inputs=[i for i in input_spaces]),
                             auto_build=False)

        inputs = input_spaces.sample()

        # Build manually, then do one step_fn for the initial staging.
        test.build()
        stage_op = test.graph_builder.api["stage"][1][0].op  # first (0) output (1) of stage API.
        test.graph_executor.monitored_session.run_step_fn(lambda step_context: step_context.session.run(
            stage_op, feed_dict={
                test.graph_builder.api["stage"][0][0].op: inputs[0],
                test.graph_builder.api["stage"][0][1].op: inputs[1],
                test.graph_builder.api["stage"][0][2].op: inputs[2],
                test.graph_builder.api["stage"][0][3].op: inputs[3],
            }
        ))

        # Unstage the inputs.
        test.test("unstage", expected_outputs=inputs)
