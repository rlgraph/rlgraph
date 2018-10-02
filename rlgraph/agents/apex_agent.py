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

from rlgraph.agents import DQNAgent


class ApexAgent(DQNAgent):
    """
    Ape-X is a DQN variant designed for large scale distributed execution where many workers
    share a distributed prioritized experience replay.

    Paper: https://arxiv.org/abs/1803.00933

    The distinction to standard DQN is mainly that Ape-X needs to provide additional operations
    to enable external updates of priorities. Ape-X also enables per default dueling and double
    DQN.
    """
    def __init__(self, memory_spec=None, **kwargs):
        """
        Args:
            memory_spec (Optional[dict,Memory]): The spec for the Memory to use for the DQN algorithm.
        """
        assert memory_spec["type"] == "prioritized_replay" or memory_spec["type"] == "mem_prioritized_replay"
        super(ApexAgent, self).__init__(memory_spec=memory_spec, huber_loss=kwargs.pop("huber_loss", True),
                                        name=kwargs.pop("name", "apex-agent"), **kwargs)

        self.num_updates = 0

    def update(self, batch=None):
        # In apex, syncing is based on num steps trained, not steps sampled.
        sync_call = None
        # Apex uses train time steps for syncing.
        self.steps_since_target_net_sync += len(batch["terminals"])
        if self.steps_since_target_net_sync >= self.update_spec["sync_interval"]:
            sync_call = "sync_target_qnet"
            self.steps_since_target_net_sync = 0
        return_ops = [0, 1]
        self.num_updates += 1
        if batch is None:
            # Add some additional return-ops to pull (left out normally for performance reasons).
            ret = self.graph_executor.execute(("update_from_memory", None, return_ops), sync_call)

            # Remove unnecessary return dicts (e.g. sync-op).
            if isinstance(ret, dict):
                ret = ret["update_from_memory"]

            if self.store_last_q_table is True:
                q_table = dict(
                    states=ret[3]["states"],
                    q_values=ret[4]
                )
                self.last_q_table = q_table

            return ret[1]
        else:
            # Add some additional return-ops to pull (left out normally for performance reasons).
            batch_input = [batch["states"], batch["actions"], batch["rewards"], batch["terminals"],
                           batch["next_states"], batch["importance_weights"]]
            ret = self.graph_executor.execute(("update_from_external_batch", batch_input), sync_call)
            # Remove unnecessary return dicts (e.g. sync-op).
            if isinstance(ret, dict):
                ret = ret["update_from_external_batch"]

            if self.store_last_q_table is True:
                q_table = dict(
                    states=batch["states"],
                    q_values=ret[3]
                )
                self.last_q_table = q_table


            # Return [1]=total loss, [2]=loss-per-item (skip [0]=update noop).
            return ret[1], ret[2]

    def get_td_loss(self, batch):
        """
        Utility method that just returns the td-loss from a batch without
        applying an update.

        Args:
            batch (dict): Input batch.

        Returns:
            Tuple: Total loss and loss per item.
        """
        batch_input = [batch["states"], batch["actions"], batch["rewards"], batch["terminals"],
                       batch["next_states"], batch["importance_weights"]]
        ret = self.graph_executor.execute(("get_td_loss", batch_input))

        # Remove unnecessary return dicts.
        if isinstance(ret, dict):
            ret = ret["get_td_loss"]

        # Return [0]=total loss, [1]=loss-per-item
        return ret[0], ret[1]

    def __repr__(self):
        return "ApexAgent"
