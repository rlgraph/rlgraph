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

from yarl.agents import DQNAgent


class ApexAgent(DQNAgent):
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
        assert memory_spec["type"] == "prioritized_replay"
        super(ApexAgent, self).__init__(discount=discount, memory_spec=memory_spec, **kwargs)

        # Apex uses train time steps for syncing.
        self.train_time_steps = 0

    def update(self, batch=None):
        # In apex, syncing is based on num steps trained, not steps sampled.
        sync_call = None
        if (self.train_time_steps - 1) % self.update_spec["sync_interval"] == 0:
            sync_call = "sync_target_qnet"

        return_ops = [0, 1]
        if batch is None:
            # Add some additional return-ops to pull (left out normally for performance reasons).
            ret = self.graph_executor.execute(("update_from_memory", None, return_ops), sync_call)

            # Remove unnecessary return dicts (e.g. sync-op).
            if isinstance(ret, dict):
                ret = ret["update_from_memory"]
        else:
            # Add some additional return-ops to pull (left out normally for performance reasons).

            batch_input = dict(
                external_batch_states=batch["states"],
                external_batch_actions=batch["actions"],
                external_batch_rewards=batch["rewards"],
                external_batch_terminals=batch["terminals"],
                external_batch_next_states=batch["next_states"]
            )
            ret = self.graph_executor.execute(("update_from_external_batch", batch_input), sync_call)

            # Remove unnecessary return dicts (e.g. sync-op).
            if isinstance(ret, dict):
                ret = ret["update_from_memory"]

        # [1]=the loss (0=update noop)
        self.train_time_steps += 1
        return ret[1]

    def __repr__(self):
        return "ApexAgent"
