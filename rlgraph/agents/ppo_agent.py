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

import copy
import numpy as np

from rlgraph.agents import Agent
from rlgraph.components import Synchronizable, Merger, Splitter,\
    Memory, PPOLossFunction, Policy
from rlgraph.spaces import Dict, IntBox, FloatBox, BoolBox


class PPOAgent(Agent):
    """
    Proximal policy optimization is a variant of policy optimization in which
    the likelihood ratio between updated and prior policy is constrained by clipping, and
    where updates are performed via repeated sub-sampling of the input batch.

    Paper: https://arxiv.org/abs/1707.06347

    """

    def __init__(self, clip_ratio, memory_spec=None, **kwargs):
        """
        Args:
            memory_spec (Optional[dict,Memory]): The spec for the Memory to use for the PPO algorithm.
        """
        super(PPOAgent, self).__init__(name=kwargs.pop("name", "ppo-agent"), **kwargs)

        self.train_time_steps = 0

        # PPO uses a ring buffer.
        self.memory = Memory.from_spec(memory_spec)
        self.record_space = Dict(states=self.state_space, actions=self.action_space, rewards=float,
                                 terminals=BoolBox(), add_batch_rank=False)

        self.policy = Policy(neural_network=self.neural_network, action_adapter_spec=None)

        self.merger = Merger(output_space=self.record_space)
        splitter_input_space = copy.deepcopy(self.record_space)
        self.splitter = Splitter(input_space=splitter_input_space)
        self.loss_function = PPOLossFunction(clip_ratio=clip_ratio, discount=self.discount)

        self.define_api_methods()
        self.build_graph()

    def _assemble_meta_graph(self, core, *params):
        # Define our interface.
        core.define_inputs("states_from_env", "external_batch_states", "external_batch_next_states",
                           "states_for_memory", space=self.state_space.with_batch_rank())
        core.define_inputs("actions_for_memory", "external_batch_actions", space=self.action_space.with_batch_rank())
        core.define_inputs("rewards_for_memory", "external_batch_rewards", space=FloatBox(add_batch_rank=True))
        core.define_inputs("terminals_for_memory", "external_batch_terminals", space=BoolBox(add_batch_rank=True))

        #core.define_inputs("deterministic", space=bool)
        core.define_inputs("time_step", space=int)
        core.define_outputs("get_actions", "insert_records",
                            "update_from_memory", "update_from_external_batch",
                            "get_batch", "get_indices", "loss")


        core.add_components(self.policy)
        # Add an Exploration for the q-net (target-net doesn't need one).
        core.add_components(self.exploration)

        # Add our Memory Component plus merger and splitter.
        core.add_components(self.memory, self.merger, self.splitter)

        # Add the loss function and optimizer.
        core.add_components(self.loss_function, self.optimizer)

        # TODO define and connect missing components

    def get_action(self, states, use_exploration=True):
        batched_states = self.state_space.force_batch(states)
        remove_batch_rank = batched_states.ndim == np.asarray(states).ndim + 1
        # Increase timesteps by the batch size (number of states in batch).
        self.timesteps += len(batched_states)
        actions = self.graph_executor.execute(
            "get_actions", inputs=dict(states_from_env=batched_states, time_step=self.timesteps)
        )

        if remove_batch_rank:
            return actions[0]
        return actions

    def get_batch(self):
        """
        Samples a batch from the priority replay memory.

        Returns:
            batch, ndarray: Sample batch and indices sampled.
        """
        batch, indices = self.graph_executor.execute("get_batch", "get")

        # Return indices so we later now which priorities to update.
        return batch, indices

    def _observe_graph(self, states, actions, internals, rewards, terminals):
        self.graph_executor.execute(
            ("insert_records", [states, actions, rewards, terminals])
        )

    def update(self, batch=None):
        if batch is None:
            _, loss = self.graph_executor.execute("update_from_memory", "loss")
        else:
            batch_input = dict(
                external_batch_states=batch["states"],
                external_batch_actions=batch["actions"],
                external_batch_rewards=batch["rewards"],
                external_batch_terminals=batch["terminals"],
                external_batch_next_states=batch["next_states"]
            )
            _, loss = self.graph_executor.execute(
                ("update_from_external_batch", batch_input),
                "loss"
            )
        self.train_time_steps += 1
        return loss

    def __repr__(self):
        return "PPOAgent()"
