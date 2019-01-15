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

from rlgraph.components.component import Component
from rlgraph.tests.dummy_components import *
from rlgraph.tests.component_test import ComponentTest


class TestGithubIssues(unittest.TestCase):
    """
    Tests issues filed on GitHub.
    """
    def test_call_in_comprehension(self):
        container = Component(scope="container")
        sub_comps = [Dummy1To1(scope="dummy-{}".format(i)) for i in range(3)]
        container.add_components(*sub_comps)

        # Define container's API:
        @rlgraph_api(name="test", component=container)
        def container_test(self_, input_):
            # results = []
            # for i in range(len(sub_comps)):
            #     results.append(sub_comps[i].run(input_))
            results = [x.run(input_) for x in sub_comps]
            return self_._graph_fn_sum(*results)

        @graph_fn(component=container)
        def _graph_fn_sum(self_, *inputs):
            return sum(inputs)

        test = ComponentTest(component=container, input_spaces=dict(input_=float))
        test.test(("test", 1.23), expected_outputs=len(sub_comps) * (1.23 + 1), decimals=2)
