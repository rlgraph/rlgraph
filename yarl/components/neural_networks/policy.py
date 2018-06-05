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
from yarl.components import Component
from yarl.components.distributions import NNOutputCleanup, Normal, Categorical
from .neural_network import NeuralNetwork


class Policy(Component):
    """
    A Policy is a Component without own graph_fns that contains a NeuralNetwork with an attached NNOutputCleanup
    followed by a Distribution Component.
    The NeuralNetwork's and the Distribution's out-Sockets are all exposed so one can extract the direct
    NN-output but also query actions (stochastically or deterministically) from the distribution.

    API:
    ins:
        input (SingleDataOp): The input to the neural network.
    outs:
        nn_output (SingleDataOp): The raw output of the neural network (before it's cleaned-up and passed through
            our action distribution).
        sample_stochastic: See Distribution component.
        sample_deterministic: See Distribution component.
        entropy: See Distribution component.
    """
    def __init__(self, neural_network, action_space, writable=True, scope="policy", **kwargs):
        """
        Args:
            neural_network (Union[NeuralNetwork,dict]): The NeuralNetwork Component or a specification dict to build
                one.
            action_space (Space): The action Space, which all actions that this Policy produces are members of.
            writable (bool): Whether the NeuralNetwork Component of this Policy is writable from another equally
                structured Policy. See Synchronizable Component for more details.
        """
        super(Policy, self).__init__(scope=scope, **kwargs)

        self.neural_network = NeuralNetwork.from_mixed(neural_network)
        self.action_space = action_space
        self.nn_cleanup = NNOutputCleanup(self.action_space)
        # The Distribution to sample (or pick) actions from.
        # Discrete action space -> Categorical distribution (each action needs a logit from network).
        if isinstance(self.action_space, IntBox):
            self.distribution = Categorical()
        # Continuous action space -> Normal distribution (each action needs mean and variance from network).
        elif isinstance(self.action_space, FloatBox):
            self.distribution = Normal()
        else:
            raise YARLError("ERROR: Space of out-Socket `action` is of type {} and not allowed in {} Component!".
                            format(type(self.action_space).__name__, self.name))

        # Define our interface.
        self.define_inputs("input")
        self.define_outputs("nn_output", "sample_deterministic", "sample_stochastic", "entropy")

        # Add NN-cleanup component and Distribution.
        self.add_components(self.nn_cleanup, self.distribution)

        # Connect everything accordingly.
        self.connect("input", (self.neural_network, "input"))
        self.connect((self.neural_network, "output"), "nn_output")
        self.connect((self.neural_network, "output"), (self.nn_cleanup, "nn_output"))
        self.connect((self.nn_cleanup, "parameters"), (self.distribution, "parameters"))
        self.connect((self.distribution, "sample_deterministic"), "sample_deterministic")
        self.connect((self.distribution, "sample_stochastic"), "sample_stochastic")
        self.connect((self.distribution, "entropy"), "entropy")

