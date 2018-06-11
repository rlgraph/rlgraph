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

import os

from yarl import distributed_backend
from yarl.agents import Agent

if distributed_backend == "ray":
    import ray


@ray.remote
class RayAgent(object):
    """
    (experimental)
    This class provides a wrapper around YARL agents so they can be used with ray's remote
    abstractions.
    """
    def __init__(self, agent_config):
        """
        Creates an agent according to the given agent config.
        Args:
            agent_config (dict): Agent config dict. Must contain a "type" key identifying the desired
                agent.
        """
        assert "type" in agent_config
        self.agent = Agent.from_spec(agent_config)

    # TODO mostly mirroring agent API here - should we inherit from agent, even if we then use
    # a child agent to execute?
    def get_action(self, states, deterministic=False):
        """
        Retrieves action(s) for the passed state(s).

        Args:
            states (Union[dict, ndarray]): State dict or array.
            deterministic (bool): If True, no exploration or sampling may be applied
                when retrieving an action.

        Returns:
             Actions dict.
        """
        return self.agent.get_action(states, deterministic)

    def get_batch(self):
        """
        Returns a batch from observed experiences according to the agent's sampling strategy.

        Returns:
            Sample dict containing the record space specified by the agent's space definitions.
        """
        # Agent must define op to return batches.
        return self.agent.call_graph_op(op="sample")

    def get_host(self):
        """
        Returns host node identifier.

        Returns:
            str: Node name this agent is running on.
        """
        return os.uname()[1]

    def call_graph_op(self, op, inputs=None):
        """
        Utility method to call any desired operation on the graph, identified via output socket.
        Delegator this call to the YARL graph executor.

        Args:
            op (str): Name of the op, i.e. the name of its output socket on the YARL metagraph.
            inputs (Optional[dict,np.array]): Dict specifying the provided inputs for some in-Sockets (key=in-Socket name,
                values=the values that should go into this Socket (e.g. numpy arrays)).
                Depending on these given inputs, the correct backend-ops can be selected within the given out-Sockets.
                If only one out-Socket is given in `sockets`, and this out-Socket only needs a single in-Socket's data,
                this in-Socket's data may be given here directly.
        Returns:
            any: Result of the op call.
        """
        self.agent.call_graph_op(op, inputs)

    def get_weights(self):
        """
        Returns the weights of this agent.

        Returns:
            any: Weight matrix and meta data
        """
        return self.agent.call_graph_op(op="get_weights")

    def set_weights(self, weights):
        """
        Returns the weights of this agent.
        """
        return self.agent.call_graph_op(op="set_weights", inputs=weights)

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
        self.agent.update(batch)



