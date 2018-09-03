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
        staging_area = StagingArea()
        input_spaces = Tuple(
            FloatBox(shape=(3, 2)),
            FloatBox(shape=(1,)),
            bool,
            IntBox(shape=(2,)),
            add_batch_rank=True
        )
        test = ComponentTest(component=staging_area, input_spaces=dict(inputs=[i for i in input_spaces]))

        inputs = input_spaces.sample(size=2)

        out = test.test(("stage", [i for i in inputs]))
        print(out)
