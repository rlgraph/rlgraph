# Copyright 2018 The YARL-Project, All Rights Reserved.
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

from yarl.agents import Agent
from yarl.components import Synchronizable, Merger, Splitter, DQNLossFunction, PrioritizedReplay, \
    Policy
from yarl.spaces import Dict, IntBox, FloatBox, BoolBox


class ApexAgent(Agent):
    """
    Ape-X is a DQN variant designed for large scale distributed execution where many workers
    share a distributed prioritized experience replay.

    Paper: https://arxiv.org/abs/1803.00933

    The distinction to standard DQN is mainly that Ape-X needs to provide additional operations
    to enable external updates of priorities. Ape-X also enables per default dueling and double
    DQN.
    """

    def __init__(self, discount=0.98, memory_spec=None, **kwargs):
        """
        Args:
            discount (float): The discount factor (gamma).
            memory_spec (Optional[dict,Memory]): The spec for the Memory to use for the DQN algorithm.
        """
        super(ApexAgent, self).__init__(**kwargs)

        self.discount = discount
        self.train_time_steps = 0

        # Apex always uses prioritized replay (not Memory.from_spec())
        self.memory = PrioritizedReplay.from_spec(memory_spec)
        self.record_space = Dict(states=self.state_space, actions=self.action_space, rewards=float,
                                 terminals=IntBox(1), add_batch_rank=False)

        # The target policy (is synced from the q-net policy every n steps).
        self.target_policy = None
        # The global copy of the q-net (if we are running in distributed mode).
        self.global_qnet = None

        # Apex always uses dueling.
        self.policy = Policy(neural_network=self.neural_network, action_adapter_spec=dict(add_dueling_layer=True))

        self.merger = Merger(output_space=self.record_space)
        splitter_input_space = copy.deepcopy(self.record_space)
        splitter_input_space["next_states"] = self.state_space
        self.splitter = Splitter(input_space=splitter_input_space)
        self.loss_function = DQNLossFunction(discount=self.discount, double_q=True)

        self.assemble_meta_graph()
        # TODO pass input space
        self.build_graph(None, self.optimizer)

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
                            "sync_target_qnet", "get_batch", "get_indices", "loss")

        # Add the Q-net, copy it (target-net) and add the target-net.
        self.target_policy = self.policy.copy(scope="target-policy")
        # Make target_policy writable
        self.target_policy.add_components(Synchronizable(), expose_apis="sync")
        core.add_components(self.policy, self.target_policy)
        # Add an Exploration for the q-net (target-net doesn't need one).
        core.add_components(self.exploration)

        # Add our Memory Component plus merger and splitter.
        core.add_components(self.memory, self.merger, self.splitter)

        # Add the loss function and optimizer.
        core.add_components(self.loss_function, self.optimizer)

        # All external/env states into preprocessor (memory already preprocessed).
        core.connect("states_from_env", (self.preprocessor_stack, "input"), label="env,s")
        core.connect("external_batch_states", (self.preprocessor_stack, "input"), label="ext,s")
        core.connect("external_batch_next_states", (self.preprocessor_stack, "input"), label="ext,sp")
        core.connect((self.preprocessor_stack, "output"), (self.policy, "nn_input"), label="s,sp")

        # Timestep into Exploration.
        core.connect("time_step", (self.exploration, "time_step"))

        # Policy output into Exploration -> into "actions".
        core.connect((self.policy, "sample_deterministic"),
                     (self.exploration, "sample_deterministic"), label="env")
        core.connect((self.policy, "sample_stochastic"),
                     (self.exploration, "sample_stochastic"), label="env")
        core.connect((self.exploration, "action"), "get_actions")
        #core.connect((self.exploration, "do_explore"), "do_explore")

        # Insert records into memory via merger.
        core.connect("states_for_memory", (self.preprocessor_stack, "input"), label="to_mem")
        core.connect((self.preprocessor_stack, "output"), (self.merger, "/states"), label="to_mem")
        for in_ in ["actions", "rewards", "terminals"]:
            core.connect(in_+"_for_memory", (self.merger, "/"+in_))
        core.connect((self.merger, "output"), (self.memory, "records"))
        core.connect((self.memory, "insert_records"), "insert_records")

        # Learn from Memory via get_batch and Splitter.
        core.connect(self.update_spec["batch_size"], (self.memory, "num_records"))
        core.connect((self.memory, "get_records"), (self.splitter, "input"), label="mem")

        # To get obtain a batch and its indices.
        core.connect((self.memory, "get_records"), "get_batch")
        core.connect((self.memory, "record_indices"), "get_indices")

        core.connect((self.splitter, "/states"), (self.policy, "nn_input"), label="mem,s")
        core.connect((self.splitter, "/actions"), (self.loss_function, "actions"))
        core.connect((self.splitter, "/rewards"), (self.loss_function, "rewards"))
        core.connect((self.splitter, "/terminals"), (self.loss_function, "terminals"))
        core.connect((self.splitter, "/next_states"), (self.target_policy, "nn_input"), label="mem,sp")
        core.connect((self.splitter, "/next_states"), (self.policy, "nn_input"), label="mem,sp")

        # Only send ext and mem labelled ops into loss function.
        q_values_socket = "q_values"
        core.connect((self.policy, q_values_socket), (self.loss_function, "q_values"), label="ext,mem,s")
        core.connect((self.target_policy, q_values_socket), (self.loss_function, "qt_values_s_"), label="ext,mem")
        core.connect((self.policy, q_values_socket), (self.loss_function, "q_values_s_"), label="ext,mem,sp")

        # Connect the Optimizer.
        core.connect((self.loss_function, "loss"), (self.optimizer, "loss"))
        core.connect((self.loss_function, "loss"), "loss")
        core.connect((self.policy, "_variables"), (self.optimizer, "vars"))
        core.connect((self.optimizer, "step"), "update_from_memory", label="mem")
        core.connect((self.optimizer, "step"), "update_from_external_batch", label="ext")

        # Connect loss to updating priority values and indices to update.
        core.connect((self.loss_function, "loss_per_item"), (self.memory, "update"))
        # TODO correct?
        core.connect((self.memory, "record_indices"), (self.memory, "indices"))

        # Add syncing capability for target-net.
        core.connect((self.policy, "_variables"), (self.target_policy, "_values"))
        core.connect((self.target_policy, "sync"), "sync_target_qnet")

    def get_action(self, states, use_exploration=True):
        batched_states = self.state_space.batched(states)
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
        batch, indices = self.graph_executor.execute(sockets=["get_batch", "get"])

        # Return indices so we later now which priorities to update.
        return batch, indices

    def update_priorities(self, indices, loss):
        """
        Updates priorities of provided indices in replay memory via externally
        provided loss.

        Args:
            indices (ndarray): Indices to update in replay memory.
            loss (ndarray):  Loss values for indices.
        """
        self.graph_executor.execute(
            sockets=["sample_indices", "sample_losses"],
            inputs=dict(sample_indices=indices, sample_losses=loss)
        )

    def _observe_graph(self, states, actions, internals, rewards, terminals):
        self.graph_executor.execute("insert_records", inputs=dict(
            states_for_memory=states,
            actions_for_memory=actions,
            rewards_for_memory=rewards,
            terminals_for_memory=terminals
        ))

    def update(self, batch=None):
        # In apex, syncing is based on num steps trained, not steps sampled.
        if (self.train_time_steps - 1) % self.update_spec["sync_interval"] == 0:
            self.graph_executor.execute("sync_target_qnet")
        if batch is None:
            _, loss = self.graph_executor.execute(["update_from_memory", "loss"])
        else:
            batch_input = dict(
                external_batch_states=batch["states"],
                external_batch_actions=batch["actions"],
                external_batch_rewards=batch["rewards"],
                external_batch_terminals=batch["terminals"],
                external_batch_next_states=batch["next_states"]
            )
            _, loss = self.graph_executor.execute(
                ["update_from_external_batch", "loss"], inputs=batch_input
            )
        self.train_time_steps += 1
        return loss

    def __repr__(self):
        return "ApexAgent"
