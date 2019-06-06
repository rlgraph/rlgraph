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

from __future__ import absolute_import, division, print_function

import logging
import unittest

from rlgraph.components.layers.nn.dense_layer import DenseLayer
from rlgraph.components.layers.strings.embedding_lookup import EmbeddingLookup
import rlgraph.spaces as spaces
from rlgraph.tests import ComponentTest
from rlgraph.tests.dummy_components import *
from rlgraph.utils import root_logger
from rlgraph.utils.visualization_util import draw_meta_graph


class TestInputSpaceChecking(unittest.TestCase):
    """
    Tests whether faulty ops are caught after calling `sanity_check_space` in `check_input_spaces` of a Component.
    """
    root_logger.setLevel(level=logging.INFO)

    def test_faulty_op_catching(self):
        """
        Adds a single component with 2-to-2 graph_fn to the core and passes two containers through it
        with flatten/split options enabled.
        """
        # Construct some easy component containing a sub-component.
        dense_layer = DenseLayer(units=2, scope="dense-layer")
        string_layer = EmbeddingLookup(embed_dim=3, vocab_size=4, scope="embed-layer")
        container_component = Component(dense_layer, string_layer)

        # Add the component's API method.
        @rlgraph_api(component=container_component)
        def test_api(self, a):
            dense_result = self.get_sub_component_by_name("dense-layer").call(a)
            # First call dense to get a vector output, then call embedding, which is expecting an int input.
            # This should fail EmbeddingLookup's input space checking (only during the build phase).
            return self.get_sub_component_by_name("embed-layer").call(dense_result)

        # Test graphviz component graph drawing.
        draw_meta_graph(container_component, apis=True)

        test = ComponentTest(
            component=container_component,
            input_spaces=dict(a=spaces.FloatBox(shape=(4,), add_batch_rank=True))
        )
