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
                 max_likelihood=True, scope="policy", **kwargs):
        """
        Args:
            network_spec (Union[NeuralNetwork,dict]): The NeuralNetwork Component or a specification dict to build
                one.

            action_space (Space): The action Space within which this Component will create actions.

            action_adapter_spec (Optional[dict]): A spec-dict to create an ActionAdapter. Use None for the default
                ActionAdapter object.

            max_likelihood (bool): Whether to pick actions according to the max-likelihood value or via sampling.
                Default: True.
        """
        super(Policy, self).__init__(scope=scope, **kwargs)

        self.neural_network = NeuralNetwork.from_spec(network_spec)
        if action_space is None:
            self.action_adapter = ActionAdapter.from_spec(action_adapter_spec)
            action_space = self.action_adapter.action_space
        else:
            self.action_adapter = ActionAdapter.from_spec(action_adapter_spec, action_space=action_space)
        self.action_space = action_space
        self.max_likelihood = max_likelihood

        # TODO: Hacky trick to implement IMPALA post-LSTM256 time-rank folding and unfolding.
        # TODO: Replace entirely via sonnet-like BatchApply Component.
        is_impala = "IMPALANetwork" in type(self.neural_network).__name__

        # Add API-method to get baseline output (if we use an extra value function baseline node).
        if isinstance(self.action_adapter, BaselineActionAdapter):
            # TODO: IMPALA attempt to speed up final pass after LSTM.
            if is_impala:
                self.time_rank_folder = ReShape(fold_time_rank=True, scope="time-rank-fold")
                self.time_rank_unfolder_v = ReShape(unfold_time_rank=True, time_major=True, scope="time-rank-unfold-v")
                self.time_rank_unfolder_a_probs = ReShape(unfold_time_rank=True, time_major=True,
                                                          scope="time-rank-unfold-a-probs")
                self.time_rank_unfolder_logits = ReShape(unfold_time_rank=True, time_major=True,
                                                         scope="time-rank-unfold-logits")
                self.time_rank_unfolder_log_probs = ReShape(unfold_time_rank=True, time_major=True,
                                                        scope="time-rank-unfold-log-probs")
                self.add_components(
                    self.time_rank_folder,
                    self.time_rank_unfolder_v,
                    self.time_rank_unfolder_a_probs,
                    self.time_rank_unfolder_log_probs,
                    self.time_rank_unfolder_logits
                )

            @rlgraph_api(component=self)
            def get_state_values_logits_probabilities_log_probs(self, nn_input, internal_states=None):
                nn_output = self.neural_network.apply(nn_input, internal_states)
                last_internal_states = nn_output.get("last_internal_states")
                nn_output = nn_output["output"]

                # TODO: IMPALA attempt to speed up final pass after LSTM.
                if is_impala:
                    nn_output = self.time_rank_folder.apply(nn_output)

                out = self.action_adapter.get_logits_probabilities_log_probs(nn_output)

                # TODO: IMPALA attempt to speed up final pass after LSTM.
                if is_impala:
                    state_values = self.time_rank_unfolder_v.apply(out["state_values"], nn_output)
                    logits = self.time_rank_unfolder_logits.apply(out["logits"], nn_output)
                    probs = self.time_rank_unfolder_a_probs.apply(out["probabilities"], nn_output)
                    log_probs = self.time_rank_unfolder_log_probs.apply(out["log_probs"], nn_output)
                else:
                    state_values = out["state_values"]
                    logits = out["logits"]
                    probs = out["probabilities"]
                    log_probs = out["log_probs"]

                return dict(state_values=state_values, logits=logits, probabilities=probs, log_probs=log_probs,
                            last_internal_states=last_internal_states)

        # Figure out our Distribution.
        if isinstance(action_space, IntBox):
            self.distribution = Categorical()
        # Continuous action space -> Normal distribution (each action needs mean and variance from network).
        elif isinstance(action_space, FloatBox):
            self.distribution = Normal()
        else:
            raise RLGraphError("ERROR: `action_space` is of type {} and not allowed in {} Component!".
                               format(type(action_space).__name__, self.name))

        self.add_components(self.neural_network, self.action_adapter, self.distribution)

        if is_impala:
            self.add_components(
                self.time_rank_folder, self.time_rank_unfolder_v, self.time_rank_unfolder_a_probs,
                self.time_rank_unfolder_log_probs, self.time_rank_unfolder_logits
            )

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
        out = self.neural_network.apply(nn_input, internal_states)
        return dict(output=out["output"], last_internal_states=out.get("last_internal_states"))

    @rlgraph_api
    def get_action(self, nn_input, internal_states=None, max_likelihood=None):
        """
        Returns an action based on NN output, action adapter output and distribution sampling.

        Args:
            nn_input (any): The input to our neural network.
            internal_states (Optional[any]): The initial internal states going into an RNN-based neural network.
            max_likelihood (Optional[bool]): If not None, use this to determine whether actions should be drawn
                from the distribution in max-likelihood or stochastic fashion.

        Returns:
            any: The drawn action.
        """
        max_likelihood = self.max_likelihood if max_likelihood is None else max_likelihood

        nn_output = self.get_nn_output(nn_input, internal_states)

        # Skip our distribution, iff discrete action-space and max-likelihood acting (greedy).
        # In that case, one does not need to create a distribution in the graph each act (only to get the argmax
        # over the logits, which is the same as the argmax over the probabilities (or log-probabilities)).
        if max_likelihood is True and isinstance(self.action_space, IntBox):
            out = self.action_adapter.get_logits_probabilities_log_probs(nn_output["output"])
            action = self._graph_fn_get_max_likelihood_action_wo_distribution(out["logits"])
        else:
            out = self.action_adapter.get_logits_probabilities_log_probs(nn_output["output"])
            action = self.distribution.draw(out["probabilities"], max_likelihood)
        return dict(action=action, last_internal_states=nn_output["last_internal_states"])

    @rlgraph_api
    def get_max_likelihood_action(self, nn_input, internal_states=None):
        """
        Args:
            nn_input (any): The input to our neural network.
            internal_states (Optional[any]): The initial internal states going into an RNN-based neural network.

        Returns:
            any: See `get_action`, but with max_likelihood force set to True.
        """
        out = self.get_logits_probabilities_log_probs(nn_input, internal_states)

        if isinstance(self.action_space, IntBox):
            action = self._graph_fn_get_max_likelihood_action_wo_distribution(out["logits"])
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
            any: See `get_action`, but with max_likelihood force set to False.
        """
        out = self.get_logits_probabilities_log_probs(nn_input, internal_states)
        action = self.distribution.sample_stochastic(out["probabilities"])
        return dict(action=action, last_internal_states=out["last_internal_states"])

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
        return dict(
            logits=aa_output["logits"], probabilities=aa_output["probabilities"], log_probs=aa_output["log_probs"],
            last_internal_states=nn_output["last_internal_states"]
        )

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
    def _graph_fn_get_max_likelihood_action_wo_distribution(self, logits):
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
