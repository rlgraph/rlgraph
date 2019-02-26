# Copyright 2018/2019 The RLgraph authors. All Rights Reserved.
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

from rlgraph.tests.agent_test import AgentTest
from rlgraph.tests.component_test import ComponentTest
from rlgraph.tests.test_util import recursive_assert_almost_equal, config_from_path
from rlgraph.tests.dummy_components import *
from rlgraph.tests.dummy_components_with_sub_components import *


__all__ = [
    "recursive_assert_almost_equal", "config_from_path",
    "AgentTest", "ComponentTest",
    # Simple dummies.
    "Dummy0To1", "Dummy1To1", "Dummy1To2", "Dummy2To1", "Dummy2GraphFns1To1",
    "DummyWithVar", "SimpleDummyWithVar", "DummyWithOptimizer",
    "FlattenSplitDummy", "NoFlattenNoSplitDummy",  "OnlyFlattenDummy",
    # Dummies with sub-components.
    "DummyWithSubComponents", "DummyCallingSubComponentsAPIFromWithinGraphFn",
    "DummyNNWithDictInput", "DummyProducingInputIncompleteBuild"
]
