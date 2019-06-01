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

import logging
import unittest

from rlgraph.agents.ppo_agent import PPOAgent
from rlgraph.environments import GridWorld
from rlgraph.tests.test_util import config_from_path
from rlgraph.utils import root_logger
from rlgraph.utils.visualization_util import draw_meta_graph


class TestVisualizations(unittest.TestCase):
    """
    Tests whether components and meta-(sub)-graphs get visualized properly.
    """
    root_logger.setLevel(level=logging.INFO)

    def test_ppo_agent_visualization(self):
        """
        Creates a PPOAgent and visualizes the root component.
        """
        env = GridWorld(world="2x2")
        ppo_agent = PPOAgent.from_spec(
            config_from_path("configs/ppo_agent_for_2x2_gridworld.json"),
            state_space=GridWorld.grid_world_2x2_flattened_state_space,
            action_space=env.action_space
        )

        # Test graphviz component-graph drawing.
        #draw_meta_graph(ppo_agent.root_component, apis=False, graph_fns=False)
        # Test graphviz component-graph w/ API drawing (only the Policy component).
        draw_meta_graph(ppo_agent.policy.neural_network, apis=True)

