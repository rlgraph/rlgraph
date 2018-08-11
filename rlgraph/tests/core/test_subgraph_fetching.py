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

import logging

from rlgraph.agents import ApexAgent
from rlgraph.environments import OpenAIGymEnv
from rlgraph.tests.test_util import config_from_path
from rlgraph.utils import root_logger
import unittest


class TestSubGraphFetching(unittest.TestCase):
    """
    Tests if the graph builder can correctly return subgraphs leading to
    a given input.
    """
    def test_subgraph_components(self):
        return
        # TODO fix when we have built selective subgraph fetching correctly.
        # Create agent.
        agent_config = config_from_path("configs/ray_apex_for_pong.json")
        agent_config["execution_spec"].pop("ray_spec")
        environment = OpenAIGymEnv("Pong-v0", frameskip=4)

        # Do not build yet.
        agent = ApexAgent.from_spec(
            agent_config, state_space=environment.state_space, action_space=environment.action_space,
            auto_build=False
        )

        # Prepare all steps until build device strategy so we can test subgraph fetching.
        agent.graph_executor.init_execution()
        agent.graph_executor.setup_graph()

        # Meta graph must be built for sub-graph tracing.
        agent.graph_builder.build_meta_graph(agent.input_spaces)

        sub_graph = agent.graph_builder.get_subgraph("update_from_external_batch")
        print("Sub graph components:")
        print(sub_graph.sub_components)
        print("Sub graph API: ")
        print(sub_graph.api_methods)

        # TODO assert which components are needed.

