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

from rlgraph import get_distributed_backend
from rlgraph.execution.ray.ray_actor import RayActor
from rlgraph.execution.ray.ray_executor import RayExecutor

if get_distributed_backend() == "ray":
    import ray


@ray.remote
class RayAgent(RayActor):
    """
    This class provides a wrapper around RLGraph Agents so they can be used with ray's remote
    abstractions.
    """
    def __init__(self, agent_config):
        """
        Creates an agent according to the given agent config.
        Args:
            agent_config (dict): Agent config dict. Must contain a "type" key identifying the desired
                agent.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing ray agent with config: {}".format(agent_config))
        assert "type" in agent_config
        self.agent = RayExecutor.build_agent_from_config(agent_config)

    def get_action(self, states, use_exploration=False):
        """
        Retrieves action(s) for the passed state(s).

        Args:
            states (Union[dict, ndarray]): State dict or array.
            use_exploration (bool): If True, no exploration or sampling may be applied
                when retrieving an action.

        Returns:
             Actions dict.
        """
        return self.agent.get_action(states, use_exploration)

    def get_batch(self):
        """
        Returns a batch from observed experiences according to the agent's sampling strategy.

        Returns:
            Sample dict containing the record space specified by the agent's space definitions.
        """
        # Agent must define op to return batches.
        return self.agent.call_api_method(op="get_records")

    def observe(self, states, actions, internals, reward, terminal):
        """
        Observes an experience tuple or a batch of experience tuples. Note: If configured,
        first uses buffers and then internally calls _observe_graph() to actually run the computation graph.
        If buffering is disabled, this just routes the call to the respective `_observe_graph()` method of the
        child Agent.

        Args:
            states (Union[dict, ndarray]): States dict or array.
            actions (Union[dict, ndarray]): Actions dict or array containing actions performed for the given state(s).
            internals (Union[list, None]): Internal state(s) returned by agent for the given states.
            reward (float): Scalar reward(s) observed.
            terminal (bool): Boolean indicating terminal.
        """
        self.agent.observe(states, actions, internals, reward, terminal)

    def update(self, batch=None):
        """
        Performs an update on the computation graph.

        Args:
            batch (Optional[dict]): Optional external data batch to use for update. If None, the
                agent should be configured to sample internally.

        Returns:
            Loss value.
        """
        return self.agent.update(batch)

    def reset(self):
        """
        Resets the wrapped Agent.
        """
        self.agent.reset()

    def call_graph_op(self, op, inputs=None):
        """
        Utility method to call any desired operation on the graph, identified via output socket.
        Delegates this call to the RLGraph graph executor.

        Args:
            op (str): Name of the op, i.e. the name of its output socket on the RLGraph metagraph.
            inputs (Optional[dict,np.array]): Dict specifying the provided api_methods for some in-Sockets (key=in-Socket name,
                values=the values that should go into this Socket (e.g. numpy arrays)).
                Depending on these given api_methods, the correct backend-ops can be selected within the given out-Sockets.
                If only one out-Socket is given in `sockets`, and this out-Socket only needs a single in-Socket's data,
                this in-Socket's data may be given here directly.

        Returns:
            any: Result of the op call.
        """
        self.agent.call_api_method(op, inputs)

    def get_policy_weights(self):
        """
        Returns the policy weights of this agent. See Agent API for docs.
        """
        return self.agent.get_policy_weights()

    def set_policy_weights(self, weights):
        """
        Sets the policy weights of this agent. See Agent API for docs.
        """
        self.agent.set_policy_weights(weights=weights)

