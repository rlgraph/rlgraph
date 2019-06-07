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
import os
import unittest

from rlgraph import rlgraph_dir
from rlgraph.agents.ppo_agent import PPOAgent
from rlgraph.environments import GridWorld
from rlgraph.tests.test_util import config_from_path
from rlgraph.utils import root_logger
from rlgraph.utils.rlgraph_errors import RLGraphError, RLGraphSpaceError
from rlgraph.utils.visualization_util import draw_meta_graph


class TestVisualizations(unittest.TestCase):
    """
    Tests whether components and meta-(sub)-graphs get visualized properly.
    """
    root_logger.setLevel(level=logging.INFO)

    def test_ppo_agent_visualization(self):
        """
        Creates a PPOAgent and visualizes meta-graph (no APIs) and the NN-component.
        """
        env = GridWorld(world="2x2")
        env.render()
        ppo_agent = PPOAgent.from_spec(
            config_from_path("configs/ppo_agent_for_2x2_gridworld.json"),
            state_space=GridWorld.grid_world_2x2_flattened_state_space,
            action_space=env.action_space
        )

        # Test graphviz component-graph drawing.
        draw_meta_graph(ppo_agent.root_component, output=rlgraph_dir + "/ppo.gv", apis=False, graph_fns=False)
        self.assertTrue(os.path.isfile(rlgraph_dir + "/ppo.gv"))
        # Test graphviz component-graph w/ API drawing (only the Policy component).
        draw_meta_graph(ppo_agent.policy.neural_network, output=rlgraph_dir + "/ppo_nn.gv", apis=True)
        self.assertTrue(os.path.isfile(rlgraph_dir + "/ppo_nn.gv"))

    def test_ppo_agent_faulty_op_visualization(self):
        """
        Creates a PPOAgent with a badly connected network and visualizes the root component.
        """
        agent_config = config_from_path("configs/ppo_agent_for_2x2_gridworld.json")
        # Sabotage the NN.
        agent_config["network_spec"] = [
            {"type": "dense", "units": 10},
            {"type": "embedding", "embed_dim": 3, "vocab_size": 4}
        ]
        env = GridWorld(world="2x2")
        # Build Agent and hence trigger the Space error.
        try:
            ppo_agent = PPOAgent.from_spec(
                agent_config,
                state_space=GridWorld.grid_world_2x2_flattened_state_space,
                action_space=env.action_space
            )
        except RLGraphSpaceError as e:
            print("Seeing expected RLGraphSpaceError ({}). Test ok.".format(e))
        else:
            raise RLGraphError("Not seeing expected RLGraphSpaceError with faulty input Space to embed layer of PPO!")
