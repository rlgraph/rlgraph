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
from yarl.components import Component, CONNECT_OUTS, CONNECT_ALL
from yarl.components.distributions import NNOutputCleanup, Normal, Categorical
from yarl.components.neural_networks.neural_network import NeuralNetwork


class Policy(Component):
    """
    A Policy is a Component without own graph_fns that contains a NeuralNetwork with an attached NNOutputCleanup
    followed by a Distribution Component.
    The NeuralNetwork's and the Distribution's out-Sockets are all exposed so one can extract the direct
    NN-output but also query actions (stochastically or deterministically) from the distribution.

    API:
    ins:
        input (SingleDataOp): The input to the neural network.
        Optional:
            sync_in (DataOpTuple): See Synchronizable Component. If writable=True.
    outs:
        nn_output (SingleDataOp): The raw output of the neural network (before it's cleaned-up and passed through
            our action distribution).
        sample_stochastic: See Distribution component.
        sample_deterministic: See Distribution component.
        entropy: See Distribution component.
        sync_out (DataOpTuple): See Synchronizable Component.
        Optional:
            sync (DataOpTuple): See Synchronizable Component. If writable=True.
    """
    def __init__(self, neural_network, scope="policy", **kwargs):
        """
        Args:
            neural_network (Union[NeuralNetwork,dict]): The NeuralNetwork Component or a specification dict to build
                one.
            #action_space (Space): The action Space, which all actions that this Policy produces are members of.
            #writable (Optional[bool]): Whether the `writable` property of the NeuralNetwork Component of this
            #    Policy should be overwritten by this value. None if the default should be used.
            #    We can only overwrite the `writable` property if `neural_network` is given as a spec dict.
            #    See Synchronizable Component for more details.
        """
        super(Policy, self).__init__(scope=scope, **kwargs)

        #if writable is not None:
        #    if not isinstance(neural_network, NeuralNetwork):
        #        neural_network["writable"] = writable
        #    else:
        #        raise YARLError("ERROR: Cannot overwrite NeuralNetwork's `writable` in constructor of {} if NN is "
        #                        "given as an already instantiated object!".format(type(self).__name__))
        self.neural_network = NeuralNetwork.from_spec(neural_network)
        self.nn_cleanup = NNOutputCleanup()
        self.distribution = None  # to be determined once we know the action Space

        # Define our interface (some of the input/output Sockets will be defined depending on the NeuralNetwork's
        # own interface, e.g. "sync_in" may be missing if the NN is not writable):
        self.define_outputs("sample_stochastic", "sample_deterministic", "entropy")

        # Add NN, NN-cleanup and Distribution Components.
        # This may additionally define the in-Socket "sync_in" and the out-Socket "sync".
        self.add_component(self.neural_network, connections=CONNECT_ALL)
        self.rename_socket("output", "nn_output")

        self.add_component(self.nn_cleanup)

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

        # Plug-in cleanup Component between NN and Distribution.
        self.connect((self.neural_network, "output"), (self.nn_cleanup, "nn_output"))
        self.connect((self.nn_cleanup, "parameters"), (self.distribution, "parameters"))
