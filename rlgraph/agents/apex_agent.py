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

import numpy as np

from rlgraph.agents import DQNAgent
from rlgraph.utils import util


class ApexAgent(DQNAgent):
    """
    Ape-X is a DQN variant designed for large scale distributed execution where many workers
    share a distributed prioritized experience replay.

    Paper: https://arxiv.org/abs/1803.00933

    The distinction to standard DQN is mainly that Ape-X needs to provide additional operations
    to enable external updates of priorities. Ape-X also enables per default dueling and double
    DQN.
    """
    def __init__(
        self,
        state_space,
        action_space,
        discount=0.98,
        preprocessing_spec=None,
        network_spec=None,
        internal_states_space=None,
        policy_spec=None,
        exploration_spec=None,
        execution_spec=None,
        optimizer_spec=None,
        observe_spec=None,
        update_spec=None,
        summary_spec=None,
        saver_spec=None,
        auto_build=True,
        name="apex-agent",
        double_q=True,
        dueling_q=True,
        huber_loss=True,
        n_step=1,
        shared_container_action_target=True,
        memory_spec=None
    ):
        """
        Args:
            state_space (Union[dict,Space]): Spec dict for the state Space or a direct Space object.
            action_space (Union[dict,Space]): Spec dict for the action Space or a direct Space object.
            preprocessing_spec (Optional[list,PreprocessorStack]): The spec list for the different necessary states
                preprocessing steps or a PreprocessorStack object itself.
            discount (float): The discount factor (gamma).
            network_spec (Optional[list,NeuralNetwork]): Spec list for a NeuralNetwork Component or the NeuralNetwork
                object itself.
            internal_states_space (Optional[Union[dict,Space]]): Spec dict for the internal-states Space or a direct
                Space object for the Space(s) of the internal (RNN) states.
            policy_spec (Optional[dict]): An optional dict for further kwargs passing into the Policy c'tor.
            exploration_spec (Optional[dict]): The spec-dict to create the Exploration Component.
            execution_spec (Optional[dict,Execution]): The spec-dict specifying execution settings.
            optimizer_spec (Optional[dict,Optimizer]): The spec-dict to create the Optimizer for this Agent.
            observe_spec (Optional[dict]): Spec-dict to specify `Agent.observe()` settings.
            update_spec (Optional[dict]): Spec-dict to specify `Agent.update()` settings.
            summary_spec (Optional[dict]): Spec-dict to specify summary settings.
            saver_spec (Optional[dict]): Spec-dict to specify saver settings.
            auto_build (Optional[bool]): If True (default), immediately builds the graph using the agent's
                graph builder. If false, users must separately call agent.build(). Useful for debugging or analyzing
                components before building.
            name (str): Some name for this Agent object.
            double_q (bool): Whether to use the double DQN loss function (see [2]).
            dueling_q (bool): Whether to use a dueling layer in the ActionAdapter  (see [3]).
            huber_loss (bool) : Whether to apply a Huber loss. (see [4]).
            n_step (Optional[int]): n-step adjustment to discounting.
            memory_spec (Optional[dict,Memory]): The spec for the Memory to use for the DQN algorithm.
        """
        assert memory_spec["type"] == "prioritized_replay" or memory_spec["type"] == "mem_prioritized_replay"
        super(ApexAgent, self).__init__(
            state_space=state_space,
            action_space=action_space,
            discount=discount,
            preprocessing_spec=preprocessing_spec,
            network_spec=network_spec,
            internal_states_space=internal_states_space,
            policy_spec=policy_spec,
            exploration_spec=exploration_spec,
            execution_spec=execution_spec,
            optimizer_spec=optimizer_spec,
            observe_spec=observe_spec,
            update_spec=update_spec,
            summary_spec=summary_spec,
            saver_spec=saver_spec,
            auto_build=auto_build,
            name=name,
            double_q=double_q,
            dueling_q=dueling_q,
            huber_loss=huber_loss,
            n_step=n_step,
            shared_container_action_target=shared_container_action_target,
            memory_spec=memory_spec
        )
        self.num_updates = 0

    def update(self, batch=None, time_percentage=None, **kwargs):
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

            return ret[1]
        else:
            # Add some additional return-ops to pull (left out normally for performance reasons).
            pps_dtype = self.preprocessed_state_space.dtype
            batch_input = [np.asarray(batch["states"], dtype=util.convert_dtype(dtype=pps_dtype, to='np')),
                           batch["actions"],
                           batch["rewards"], batch["terminals"],
                           np.asarray(batch["next_states"], dtype=util.convert_dtype(dtype=pps_dtype, to='np')),
                           batch["importance_weights"],
                           True]
            ret = self.graph_executor.execute(("update_from_external_batch", batch_input), sync_call)
            # Remove unnecessary return dicts (e.g. sync-op).
            if isinstance(ret, dict):
                ret = ret["update_from_external_batch"]

            # Return [1]=total loss, [2]=loss-per-item (skip [0]=update noop).
            return ret[1], ret[2]

    def __repr__(self):
        return "ApexAgent"
