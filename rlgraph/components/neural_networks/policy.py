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

from rlgraph import get_backend
from rlgraph.utils.rlgraph_errors import RLGraphError
from rlgraph.spaces import IntBox, FloatBox
from rlgraph.components.component import Component
from rlgraph.components.distributions import Normal, Categorical
from rlgraph.components.neural_networks.neural_network import NeuralNetwork
from rlgraph.components.action_adapters.action_adapter import ActionAdapter
from rlgraph.components.action_adapters.baseline_action_adapter import BaselineActionAdapter
from rlgraph.components.layers.preprocessing.reshape import ReShape
from rlgraph.utils.decorators import rlgraph_api, graph_fn

if get_backend() == "tf":
    import tensorflow as tf
elif get_backend() == "pytorch":
    import torch


class Policy(Component):
    """
    A Policy is a wrapper Component that contains a NeuralNetwork, an ActionAdapter and a Distribution Component.
    """
    def __init__(self, network_spec, action_space=None, action_adapter_spec=None,
                 deterministic=True, batch_apply=False, batch_apply_action_adapter=False,
                 scope="policy", **kwargs):
        """
        Args:
            network_spec (Union[NeuralNetwork,dict]): The NeuralNetwork Component or a specification dict to build
                one.

            action_space (Space): The action Space within which this Component will create actions.

            action_adapter_spec (Optional[dict]): A spec-dict to create an ActionAdapter. Use None for the default
                ActionAdapter object.

            deterministic (bool): Whether to pick actions according to the max-likelihood value or via sampling.
                Default: True.

            batch_apply (bool): Whether to wrap both the NN and the ActionAdapter with a BatchApply Component in order
                to fold time rank into batch rank before a forward pass.
                Note that only one of `batch_apply` or `batch_apply_action_adapter` may be True.
                Default: False.

            batch_apply_action_adapter (bool): Whether to wrap only the ActionAdapter with a BatchApply Component in
                order to fold time rank into batch rank before a forward pass.
                Note that only one of `batch_apply` or `batch_apply_action_adapter` may be True.
                Default: False.
        """
        super(Policy, self).__init__(scope=scope, **kwargs)

        self.batch_apply = batch_apply
        self.batch_apply_action_adapter = batch_apply_action_adapter
        assert self.batch_apply is False or self.batch_apply_action_adapter is False,\
            "ERROR: Either one of `batch_apply` or `batch_apply_action_adapter` must be False!"

        # Do manual folding and unfolding as not to have to wrap too many components into a BatchApply.
        self.folder = None
        self.unfolder = None
        if self.batch_apply is True:
            self.folder = ReShape(fold_time_rank=True, scope="time-rank-folder")
            self.unfolder = ReShape(unfold_time_rank=True, scope="time-rank-unfolder")

        self.neural_network = NeuralNetwork.from_spec(network_spec)  # type: NeuralNetwork

        if action_space is None:
            self.action_adapter = ActionAdapter.from_spec(
                action_adapter_spec, batch_apply=self.batch_apply_action_adapter
            )
            action_space = self.action_adapter.action_space
        else:
            self.action_adapter = ActionAdapter.from_spec(
                action_adapter_spec, action_space=action_space, batch_apply=self.batch_apply_action_adapter
            )
        self.action_space = action_space
        self.deterministic = deterministic

        # Add API-method to get baseline output (if we use an extra value function baseline node).
        if isinstance(self.action_adapter, BaselineActionAdapter):

            @rlgraph_api(component=self)
            def get_state_values_logits_probabilities_log_probs(self, nn_input, internal_states=None):
                nn_output = self.get_nn_output(nn_input, internal_states)
                out = self.action_adapter.get_logits_probabilities_log_probs(nn_output["output"])

                state_values = out["state_values"]
                logits = out["logits"]
                probs = out["probabilities"]
                log_probs = out["log_probs"]

                if self.batch_apply is True:
                    state_values = self.unfolder.apply(state_values, nn_input)
                    logits = self.unfolder.apply(logits, nn_input)
                    probs = self.unfolder.apply(probs, nn_input)
                    log_probs = self.unfolder.apply(log_probs, nn_input)

                return dict(state_values=state_values, logits=logits, probabilities=probs, log_probs=log_probs,
                            last_internal_states=nn_output.get("last_internal_states"))

        # Figure out our Distribution.
        if isinstance(action_space, IntBox):
            self.distribution = Categorical()
        # Continuous action space -> Normal distribution (each action needs mean and variance from network).
        elif isinstance(action_space, FloatBox):
            self.distribution = Normal()
        else:
            raise RLGraphError("ERROR: `action_space` is of type {} and not allowed in {} Component!".
                               format(type(action_space).__name__, self.name))

        self.add_components(self.neural_network, self.action_adapter, self.distribution, self.folder, self.unfolder)

    # Define our interface.
    @rlgraph_api
    def get_nn_output(self, nn_input, internal_states=None):
        """
        Args:
            nn_input (any): The input to our neural network.
            internal_states (Optional[any]): The initial internal states going into an RNN-based neural network.

        Returns:
            any: The raw output of the neural network (before it's cleaned-up and passed through the ActionAdapter).
        """
        if self.batch_apply is True:
            nn_input = self.folder.apply(nn_input)

        out = self.neural_network.apply(nn_input, internal_states)
        return dict(output=out["output"], last_internal_states=out.get("last_internal_states"))

    @rlgraph_api
    def get_action_layer_output(self, nn_input, internal_states=None):
        """
        Args:
            nn_input (any): The input to our neural network.
            internal_states (Optional[any]): The initial internal states going into an RNN-based neural network.

        Returns:
            any: The raw output of the action layer of the ActionAdapter (including possibly the last internal states
                of a RNN-based NN).
        """
        nn_output = self.get_nn_output(nn_input, internal_states)
        action_layer_output = self.action_adapter.get_action_layer_output(nn_output["output"])
        # Add last internal states to return value.
        return dict(output=action_layer_output["output"], last_internal_states=nn_output["last_internal_states"])

    @rlgraph_api
    def get_logits_probabilities_log_probs(self, nn_input, internal_states=None):
        """
        Args:
            nn_input (any): The input to our neural network.
            internal_states (Optional[any]): The initial internal states going into an RNN-based neural network.

        Returns:
            Dict:
                logits: The (reshaped) logits from the ActionAdapter.
                probabilities: The probabilities gained from the softmaxed logits.
                log_probs: The log(probabilities) values.
        """
        nn_output = self.get_nn_output(nn_input, internal_states)
        aa_output = self.action_adapter.get_logits_probabilities_log_probs(nn_output["output"])

        logits, probabilities, log_probs = aa_output["logits"], aa_output["probabilities"], aa_output["log_probs"]

        if self.batch_apply is True:
            logits = self.unfolder.apply(logits, nn_input)
            probabilities = self.unfolder.apply(probabilities, nn_input)
            log_probs = self.unfolder.apply(log_probs, nn_input)

        return dict(
            logits=logits, probabilities=probabilities, log_probs=log_probs,
            last_internal_states=nn_output["last_internal_states"]
        )

    @rlgraph_api
    def get_action(self, nn_input, deterministic=None, internal_states=None):
        """
        Returns an action based on NN output, action adapter output and distribution sampling.

        Args:
            nn_input (any): The input to our neural network.
            deterministic (Optional[bool]): If not None, use this to determine whether actions should be drawn
                from the distribution in max-likelihood (deterministic) or stochastic fashion.
            internal_states (Optional[any]): The initial internal states going into an RNN-based neural network.

        Returns:
            any: The drawn action.
        """
        deterministic = self.deterministic if deterministic is None else deterministic

        out = self.get_logits_probabilities_log_probs(nn_input, internal_states)

        # Skip our distribution, iff discrete action-space and deterministic acting (greedy).
        # In that case, one does not need to create a distribution in the graph each act (only to get the argmax
        # over the logits, which is the same as the argmax over the probabilities (or log-probabilities)).
        if deterministic is True and isinstance(self.action_space, IntBox):
            action = self._graph_fn_get_deterministic_action_wo_distribution(out["logits"])
        else:
            action = self.distribution.draw(out["probabilities"], deterministic)

        return dict(action=action, last_internal_states=out["last_internal_states"])

    @rlgraph_api
    def get_action_log_probs(self, nn_input, actions, internal_states=None):
        """
        Computes the log-likelihood for a given set of actions under the distribution induced by a set of states.

        Args:
            nn_input (any): The input to our neural network.
            actions (any): The actions for which to get log-probs returned.

            internal_states (Optional[any]): The initial internal states going into an RNN-based neural network.

        Returns:
            Log-probs of actions under current policy
        """
        out = self.get_logits_probabilities_log_probs(nn_input, internal_states)

        # Probabilities under current action.
        action_log_probs = self.distribution.log_prob(out["probabilities"], actions)

        return dict(action_log_probs=action_log_probs, logits=out["logits"])

    @rlgraph_api
    def get_deterministic_action(self, nn_input, internal_states=None):
        """
        Args:
            nn_input (any): The input to our neural network.
            internal_states (Optional[any]): The initial internal states going into an RNN-based neural network.

        Returns:
            any: See `get_action`, but with deterministic force set to True.
        """
        out = self.get_logits_probabilities_log_probs(nn_input, internal_states)

        if isinstance(self.action_space, IntBox):
            action = self._graph_fn_get_deterministic_action_wo_distribution(out["logits"])
        else:
            action = self.distribution.sample_deterministic(out["probabilities"])

        return dict(action=action, last_internal_states=out["last_internal_states"])

    @rlgraph_api
    def get_stochastic_action(self, nn_input, internal_states=None):
        """
        Args:
            nn_input (any): The input to our neural network.
            internal_states (Optional[any]): The initial internal states going into an RNN-based neural network.

        Returns:
            any: See `get_action`, but with deterministic force set to False.
        """
        out = self.get_logits_probabilities_log_probs(nn_input, internal_states)
        action = self.distribution.sample_stochastic(out["probabilities"])

        return dict(action=action, last_internal_states=out["last_internal_states"])

    @rlgraph_api
    def get_entropy(self, nn_input, internal_states=None):
        """
        Args:
            nn_input (any): The input to our neural network.
            internal_states (Optional[any]): The initial internal states going into an RNN-based neural network.

        Returns:
            any: See Distribution component.
        """
        out = self.get_logits_probabilities_log_probs(nn_input, internal_states)
        entropy = self.distribution.entropy(out["probabilities"])

        return dict(entropy=entropy, last_internal_states=out["last_internal_states"])

    @graph_fn
    def _graph_fn_get_deterministic_action_wo_distribution(self, logits):
        """
        Use this function only for discrete action spaces to circumvent using a full-blown
        backend-specific distribution object (e.g. tf.distribution.Multinomial).

        Args:
            logits (SingleDataOp): Logits over which to pick the argmax (greedy action).

        Returns:
            SingleDataOp: The argmax over the last rank of the input logits.
        """
        if get_backend() == "tf":
            return tf.argmax(logits, axis=-1, output_type=tf.int32)
        elif get_backend() == "pytorch":
            return torch.argmax(logits, dim=-1).int()
