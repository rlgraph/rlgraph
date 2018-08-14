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

from rlgraph import RLGraphError, get_backend
from rlgraph.spaces import IntBox, FloatBox
from rlgraph.components.component import Component
from rlgraph.components.common.synchronizable import Synchronizable
from rlgraph.components.distributions import Normal, Categorical
from rlgraph.components.neural_networks.neural_network import NeuralNetwork
from rlgraph.components.action_adapters.action_adapter import ActionAdapter
from rlgraph.components.action_adapters.dueling_action_adapter import DuelingActionAdapter
from rlgraph.components.action_adapters.baseline_action_adapter import BaselineActionAdapter

if get_backend() == "tf":
    import tensorflow as tf


class Policy(Component):
    """
    A Policy is a Component without own graph_fns that contains a NeuralNetwork with an attached ActionAdapter
    followed by a Distribution Component.

    API:
        get_action(nn_input, internals) -> (action, last_internals).
            Single action based on the neural network input AND `self.max_likelihood`.
            If True, returns a deterministic (max_likelihood) sample, if False, returns a stochastic sample.
        get_nn_output(nn_input, internals): The raw output of the neural network (before it's cleaned-up and passed through
            our ActionAdapter).
        get_action_layer_output(nn_input, internals) (SingleDataOp): The raw output of the action layer of the
            ActionAdapter.
        get_q_values(nn_input, internals): The Q-values (action-space shaped) as calculated by the action-adapter.
        get_logits_parameters_log_probs (nn_input, internals): See ActionAdapter Component.
            action_layer_output_reshaped (SingleDataOp): The action layer output, reshaped according to the action
                    space.
        get_max_likelihood_action: See get_action, but with max_likelihood force set to True.
        get_stochastic_action: See get_action, but with max_likelihood force set to False.
        entropy: See Distribution component.

        Optional:
            # TODO: Fix this and automatically forward all action adapter's API methods (with the preceding NN call)
            # TODO: to the policy.
            If action_adapter is a DuelingActionAdapter:
                get_dueling_output:
                    state_value (SingleDataOp): The state value diverged from the first output node of the previous
                        layer.
                    advantage_values (SingleDataOp): The advantage values (already reshaped) for the different actions.
                    q_values (SingleDataOp): The Q-values (already reshaped) for the different state-action pairs.
                        Calculated according to the dueling layer logic.
            If action_adapter is a BaselineActionAdapter:
                get_baseline_output:
                    state_value (SingleDataOp): The state value diverged from the first output node of the previous
                        layer.
                    logits (SingleDataOp): The (action-space reshaped, but otherwise raw) logits.
    """
    def __init__(self, neural_network, action_space=None,
                 writable=False, action_adapter_spec=None, max_likelihood=True, scope="policy", **kwargs):
        """
        Args:
            neural_network (Union[NeuralNetwork,dict]): The NeuralNetwork Component or a specification dict to build
                one.
            action_space (Space): The action Space within which this Component will create actions.
            writable (bool): Whether this Policy can be synced to by another (equally structured) Policy.
                Default: False.
            action_adapter_spec (Optional[dict]): A spec-dict to create an ActionAdapter. Use None for the default
                ActionAdapter object.
            max_likelihood (bool): Whether to pick actions according to the max-likelihood value or via sampling.
                Default: True.
        """
        super(Policy, self).__init__(scope=scope, **kwargs)

        self.neural_network = NeuralNetwork.from_spec(neural_network)
        self.writable = writable
        if action_space is None:
            self.action_adapter = ActionAdapter.from_spec(action_adapter_spec)
            action_space = self.action_adapter.action_space
        else:
            self.action_adapter = ActionAdapter.from_spec(action_adapter_spec, action_space=action_space)
        self.action_space = action_space
        self.max_likelihood = max_likelihood

        # Add API-method to get dueling output (if we use a dueling layer).
        if isinstance(self.action_adapter, DuelingActionAdapter):
            def get_dueling_output(self_, nn_input, internals=None):
                nn_output, internals = self_.unify_nn_output(self_.call(self.neural_network.apply, nn_input, internals))
                state_value, advantages, q_values = self_.call(self_.action_adapter.get_dueling_output, nn_output)
                return (state_value, advantages, q_values, internals) if internals is not None else \
                    (state_value, advantages, q_values)

            self.define_api_method("get_dueling_output", get_dueling_output)

            def get_q_values(self_, nn_input, internals=None):
                nn_output, internals = self_.unify_nn_output(
                    self_.call(self_.neural_network.apply, nn_input, internals)
                )
                _, _, q = self_.call(self_.action_adapter.get_dueling_output, nn_output)
                return (q, internals) if internals is not None else q
        # Add API-method to get baseline output (if we use an extra value function baseline node).
        elif isinstance(self.action_adapter, BaselineActionAdapter):
            def get_baseline_output(self_, nn_input, internals=None):
                nn_output, internals = self_.unify_nn_output(
                    self_.call(self_.neural_network.apply, nn_input, internals)
                )
                state_value, logits = self_.call(self_.action_adapter.get_state_values_and_logits, nn_output)
                return (state_value, logits, internals) if internals is not None else (state_value, logits)

            self.define_api_method("get_baseline_output", get_baseline_output)

            def get_q_values(self_, nn_input, internals=None):
                nn_output, internals = self_.unify_nn_output(
                    self_.call(self_.neural_network.apply, nn_input, internals)
                )
                _, logits = self_.call(self_.action_adapter.get_state_values_and_logits, nn_output)
                return (logits, internals) if internals is not None else logits
        else:
            def get_q_values(self_, nn_input, internals=None):
                nn_output, internals = self_.unify_nn_output(
                    self_.call(self_.neural_network.apply, nn_input, internals)
                )
                logits, _, _ = self_.call(self_.action_adapter.get_logits_parameters_log_probs, nn_output)
                return (logits, internals) if internals is not None else logits

        self.define_api_method("get_q_values", get_q_values)

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

        # Add Synchronizable API to ours.
        if self.writable:
            self.add_components(Synchronizable(), expose_apis="sync")

    # Define our interface.
    def get_action(self, nn_input, internals=None):
        nn_output, internals = self.unify_nn_output(self.call(self.neural_network.apply, nn_input, internals))
        # TEST without distribution iff discrete action-space and max-likelihood acting (greedy).
        # In that case, one does not need to create a distribution in the graph each act (only to get the max anyway
        # over the logits, which is the same as the max over the probabilities/log-probabilities).
        if self.max_likelihood is True and isinstance(self.action_space, IntBox):
            logits, _, _ = self.call(self.action_adapter.get_logits_parameters_log_probs, nn_output)
            sample = self.call(self._graph_fn_get_max_likelihood_action_wo_distribution, logits)
        else:
            _, parameters, _ = self.call(self.action_adapter.get_logits_parameters_log_probs, nn_output)
            sample = self.call(self.distribution.draw, parameters, self.max_likelihood)
        return (sample, internals) if internals is not None else sample

    def get_nn_output(self, nn_input, internals=None):
        nn_output, internals = self.unify_nn_output(self.call(self.neural_network.apply, nn_input, internals))
        return (nn_output, internals) if internals is not None else nn_output

    def get_action_layer_output(self, nn_input, internals=None):
        nn_output, internals = self.unify_nn_output(self.call(self.neural_network.apply, nn_input, internals))
        action_layer_output = self.call(self.action_adapter.get_action_layer_output, nn_output)
        return (action_layer_output, internals) if internals is not None else action_layer_output

    def get_logits_parameters_log_probs(self, nn_input, internals=None):
        nn_output, internals = self.unify_nn_output(self.call(self.neural_network.apply, nn_input, internals))
        logits, parameters, log_probs = self.call(self.action_adapter.get_logits_parameters_log_probs, nn_output)
        return (logits, parameters, log_probs, internals) if internals is not None else (logits, parameters, log_probs)

    def get_entropy(self, nn_input, internals=None):
        nn_output, internals = self.unify_nn_output(self.call(self.neural_network.apply, nn_input, internals))
        _, parameters, _ = self.call(self.action_adapter.get_logits_parameters_log_probs, nn_output)
        entropy = self.call(self.distribution.entropy, parameters)
        return (entropy, internals) if internals is not None else entropy

    def get_max_likelihood_action(self, nn_input, internals=None):
        nn_output, internals = self.unify_nn_output(self.call(self.neural_network.apply, nn_input, internals))
        _, parameters, _ = self.call(self.action_adapter.get_logits_parameters_log_probs, nn_output)
        sample = self.call(self.distribution.sample_deterministic, parameters)
        return (sample, internals) if internals is not None else sample

    def get_stochastic_action(self, nn_input, internals=None):
        nn_output, internals = self.unify_nn_output(self.call(self.neural_network.apply, nn_input, internals))
        _, parameters, _ = self.call(self.action_adapter.get_logits_parameters_log_probs, nn_output)
        sample = self.call(self.distribution.sample_stochastic, parameters)
        return (sample, internals) if internals is not None else sample

    @staticmethod
    def unify_nn_output(nn_output):
        # Neural net returns output and internal-state.
        if isinstance(nn_output, tuple) and len(nn_output) == 2:
            return nn_output[0], nn_output[1]
        else:
            return nn_output, None

    def _graph_fn_get_max_likelihood_action_wo_distribution(self, parameters):
        if get_backend() == "tf":
            return tf.argmax(parameters, axis=-1, output_type=tf.int32)
