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

from yarl import YARLError
from yarl.spaces import IntBox, FloatBox
from yarl.components import Component, Synchronizable
from yarl.components.distributions import Normal, Categorical
from yarl.components.neural_networks.neural_network import NeuralNetwork
from yarl.components.neural_networks.action_adapter import ActionAdapter


class Policy(Component):
    """
    A Policy is a Component without own graph_fns that contains a NeuralNetwork with an attached ActionAdapter
    followed by a Distribution Component.
    The NeuralNetwork's and the Distribution's out-Sockets are all exposed so one can extract the direct
    NN-output but also query actions (stochastically or deterministically) from the distribution.

    API:
    ins:
        nn_input (SingleDataOp): The input to the neural network.
    outs:
        nn_output (SingleDataOp): The raw output of the neural network (before it's cleaned-up and passed through
            our ActionAdapter).
        action_layer_output (SingleDataOp): The flat output of the action layer of the ActionAdapter.
        Optional:
            If action_adapter has a DuelingLayer:
                state_value (SingleDataOp): The state value diverged from the first output node of the previous layer.
                advantage_values (SingleDataOp): The advantage values (already reshaped) for the different actions.
                q_values (SingleDataOp): The Q-values (already reshaped) for the different state-action pairs.
                    Calculated according to the dueling layer logic.
            else:
                action_layer_output_reshaped (SingleDataOp): The action layer output, reshaped according to the action
                    space.
        parameters (SingleDataOp): The softmaxed action_layer_outputs (probability parameters) going into the
            Distribution Component.
        logits (SingleDataOp): The logs of the parameter (probability) values.
        sample_stochastic: See Distribution component.
        sample_deterministic: See Distribution component.
        entropy: See Distribution component.
    """
    def __init__(self, neural_network, writable=False, action_adapter_spec=None, scope="policy", **kwargs):
        """
        Args:
            neural_network (Union[NeuralNetwork,dict]): The NeuralNetwork Component or a specification dict to build
                one.
            writable (bool): Whether this Policy can be synced to by another (equally structured) Policy.
                Default: False.
            action_adapter_spec (Optional[dict]): A spec-dict to create an ActionAdapter. USe None for the default
                ActionAdapter object.
        """
        super(Policy, self).__init__(scope=scope, **kwargs)

        self.neural_network = NeuralNetwork.from_spec(neural_network)
        self.writable = writable
        self.action_adapter = ActionAdapter.from_spec(action_adapter_spec)
        self.distribution = None  # to be determined once we know the action Space

        # Define our interface (some of the input/output Sockets will be defined depending on the NeuralNetwork's
        # own interface, e.g. "sync_in" may be missing if the NN is not writable):
        self.define_outputs("parameters", "logits", "sample_stochastic", "sample_deterministic", "entropy")

        # Add NN, connect it through and then rename its "output" into Policy's "nn_output".
        self.add_component(self.neural_network, connections=CONNECT_ALL)
        self.rename_socket("input", "nn_input")
        self.rename_socket("output", "nn_output")

        # Add the Adapter, connect the network's "output" into it and the "logits" Socket.
        self.add_component(self.action_adapter, connections=CONNECT_OUTS)
        self.connect((self.neural_network, "output"), (self.action_adapter, "nn_output"))
        #self.connect((self.action_adapter, "action_layer_output"), "action_layer_output")
        #self.connect((self.action_adapter, "parameters"), "parameters")
        #self.connect((self.action_adapter, "logits"), "logits")

        # Add Synchronizable API to ours.
        if self.writable:
            self.add_component(Synchronizable(), connections=CONNECT_ALL)

    def check_input_spaces(self, input_spaces, action_space):
        # The Distribution to sample (or pick) actions from.
        # Discrete action space -> Categorical distribution (each action needs a logit from network).
        if isinstance(action_space, IntBox):
            self.distribution = Categorical()
        # Continuous action space -> Normal distribution (each action needs mean and variance from network).
        elif isinstance(action_space, FloatBox):
            self.distribution = Normal()
        else:
            raise YARLError("ERROR: Space of out-Socket `action` is of type {} and not allowed in {} Component!".
                            format(type(action_space).__name__, self.name))

        # This defines out-Sockets "sample_stochastic/sample_deterministic/entropy".
        self.add_component(self.distribution, connections=CONNECT_OUTS)
        # Plug-in Adapter Component into Distribution.
        self.connect((self.action_adapter, "parameters"), (self.distribution, "parameters"))
