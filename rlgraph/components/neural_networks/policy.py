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

from rlgraph import RLGraphError
from rlgraph.spaces import IntBox, FloatBox
from rlgraph.components import Component, Synchronizable
from rlgraph.components.distributions import Normal, Categorical
from rlgraph.components.neural_networks.neural_network import NeuralNetwork
from rlgraph.components.neural_networks.action_adapter import ActionAdapter


class Policy(Component):
    """
    A Policy is a Component without own graph_fns that contains a NeuralNetwork with an attached ActionAdapter
    followed by a Distribution Component.
    The NeuralNetwork's and the Distribution's out-Sockets are all exposed so one can extract the direct
    NN-output but also query actions (stochastically or deterministically) from the distribution.

    API:
        get_nn_output(nn_input): The raw output of the neural network (before it's cleaned-up and passed through
            our ActionAdapter).
        get_action_layer_output(nn_input) (SingleDataOp): The flat output of the action layer of the ActionAdapter.
        Optional:
            If action_adapter has a DuelingLayer:
                state_value (SingleDataOp): The state value diverged from the first output node of the previous layer.
                advantage_values (SingleDataOp): The advantage values (already reshaped) for the different actions.
                q_values (SingleDataOp): The Q-values (already reshaped) for the different state-action pairs.
                    Calculated according to the dueling layer logic.
            else:
                action_layer_output_reshaped (SingleDataOp): The action layer output, reshaped according to the action
                    space.
        get_logits_parameters_log_probs: See ActionAdapter Component.
        sample_stochastic: See Distribution component.
        sample_deterministic: See Distribution component.
        entropy: See Distribution component.
    """
    def __init__(self, neural_network, action_space=None,
                 writable=False, action_adapter_spec=None, scope="policy", **kwargs):
        """
        Args:
            neural_network (Union[NeuralNetwork,dict]): The NeuralNetwork Component or a specification dict to build
                one.
            action_space (Space): The action Space within which this Component will create actions.
            writable (bool): Whether this Policy can be synced to by another (equally structured) Policy.
                Default: False.
            action_adapter_spec (Optional[dict]): A spec-dict to create an ActionAdapter. USe None for the default
                ActionAdapter object.
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

        # Add API-method to get dueling output (if we use a dueling layer).
        if self.action_adapter.add_dueling_layer:
            def get_dueling_output(self_, nn_input):
                nn_output = self_.call(self_.neural_network.apply, nn_input)
                return self_.call(self_.action_adapter.get_dueling_output, nn_output)

            self.define_api_method("get_dueling_output", get_dueling_output)

            def get_q_values(self_, nn_input):
                nn_output = self_.call(self_.neural_network.apply, nn_input)
                _, _, q = self_.call(self_.action_adapter.get_dueling_output, nn_output)
                return q
        else:
            def get_q_values(self_, nn_input):
                nn_output = self_.call(self_.neural_network.apply, nn_input)
                logits, _, _ = self_.call(self_.action_adapter.get_logits_parameters_log_probs, nn_output)
                return logits

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
    def get_nn_output(self, nn_input):
        nn_output = self.call(self.neural_network.apply, nn_input)
        return nn_output

    def get_action_layer_output(self, nn_input):
        nn_output = self.call(self.neural_network.apply, nn_input)
        action_layer_output = self.call(self.action_adapter.get_action_layer_output, nn_output)
        return action_layer_output

    def get_logits_parameters_log_probs(self, nn_input):
        nn_output = self.call(self.neural_network.apply, nn_input)
        logits, parameters, log_probs = self.call(self.action_adapter.get_logits_parameters_log_probs, nn_output)
        return logits, parameters, log_probs

    def get_entropy(self, nn_input):
        nn_output = self.call(self.neural_network.apply, nn_input)
        _, parameters, _ = self.call(self.action_adapter.get_logits_parameters_log_probs, nn_output)
        entropy = self.call(self.distribution.entropy, parameters)
        return entropy

    def sample_stochastic(self, nn_input):
        nn_output = self.call(self.neural_network.apply, nn_input)
        _, parameters, _ = self.call(self.action_adapter.get_logits_parameters_log_probs, nn_output)
        sample = self.call(self.distribution.sample_stochastic, parameters)
        return sample

    def sample_deterministic(self, nn_input):
        nn_output = self.call(self.neural_network.apply, nn_input)
        _, parameters, _ = self.call(self.action_adapter.get_logits_parameters_log_probs, nn_output)
        sample = self.call(self.distribution.sample_deterministic, parameters)
        return sample

