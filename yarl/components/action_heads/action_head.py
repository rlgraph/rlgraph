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

import tensorflow as tf

from yarl.components import Component, CONNECT_INS
from yarl.components.distributions import Bernoulli


class ActionHeadComponent(Component):
    def __init__(self, action_space, add_softmax=False, picking_strategy="max-likelihood", epsilon_component=None,
                 noise_spec=None, scope="action-head", **kwargs):
        """
        Args:
            action_space (Space): The action Space. Outputs of the "action" Socket must be members of this Space.
            add_softmax (bool): Whether to softmax the NN outputs (before further processing them).
            picking_strategy (str): One of: "max-likelihood", "random", "sample".
            epsilon_component (Component): The sub-Component for epsilon-based exploration (e.g. a DecayComponent).
            #noise_spec (dict): The specification dict for a noise generator that adds noise to the NN's output.
        """
        super(ActionHeadComponent, self).__init__(scope=scope, **kwargs)

        self.add_softmax = add_softmax

        self.action_space = action_space
        assert self.action_space.has_batch_rank, "ERROR: `action_space` does not have batch rank!"

        self.picking_strategy = picking_strategy
        self.epsilon_component = epsilon_component
        self.noise_spec = noise_spec

        # The Bernoulli Distribution to check, whether to do (epsilon) exploration or not.
        self.epsilon_component = Bernoulli()
        # The Distribution to sample (or pick) actions from.
        self.action_distribution = None

        # We only need the global timestep and the NN output to determine an action.
        self.define_inputs("time_step", "nn_output")
        self.define_outputs("action")

        # Add epsilon Component and connect accordingly.
        if self.epsilon_component is not None:
            self.add_component(self.epsilon_component, connect=CONNECT_INS)
            #self.connect(self.epsilon_component.get_output("do_explore"), "epsilon")
            #self.add_computation(["time_step", "nn_output", self.epsilon_component.get_output("epsilon")],
            #                     "action", self._computation_act)
        # Else, set epsilon to 0.0 (no epsilon-based exploration).
        else:
            self.add_computation(["time_step", "nn_output", 0.0], "action", self._computation_act)

    def create_variables(self, input_spaces):
        flat_action_space = input_spaces["nn_output"]
        assert flat_action_space.flat_dim == self.action_space.flat_dim, \
            "ERROR: The flat_dims of incoming NN-output ({}) and our action_space ({}) don't match!". \
            format(flat_action_space.flat_dim, self.action_space.flat_dim)

        #if self.epsilon_component is not None:
        # Now that we know the NN's output shape, we can generate our distributions.
        #if self.epsilon_component is not None:
            #self.epsilon_distribution

    def _computation_act(self, time_step, nn_output, epsilon):
        """
        Args:
            time_step (DataOp): The time-step information needed to figure out the amount and nature of exploration.
            nn_output (DataOp): The output from the neural network (policy, Q-net, etc..).

        Returns:
            DataOp: The DataOp representing the action. This will match the shape of self.action_space.
        """
        # Reshape NN-output.
        reshaped_nn_output = tf.reshape(nn_output, self.action_space.shape_with_batch_rank)

        # Pass NN-output through softmax first?
        if self.add_softmax:
            nn_output = tf.nn.softmax(nn_output)  # axis -1 ok?

        # Sample our epsilon-defined Bernoulli distribution.
        do_explore = False
        if self.epsilon_component is not None:
            pass
            #do_explore =



"""        
        ActionHeadComponent
    __init__(action-space, epsilon-comp=epsilon-component if None, no epsilon;
        alternatively: noise-component to be added to continuous action output?
        epsilon-strategy=[greedy|sample|random])
    in: network-output (q-values or probs), time-step
    out: "act"
    _computation_act: use sub-exploration-component with connected  time-step to calculate epsilon:
        if explore: pick random action from some distribution (needs to be specified in c'tor?).
        else: pick greedy action (or some other strategy) according to network-output.


Possibilities:
    - discrete output (q-values (Q-Learning) or (softmaxed) discrete-action probs (PG))
        makes sense to use epsilon-style exploration (if epsilon: random, else greedy or prob-based-sample).
    - continuous output (params for a distribution to pick actions from)
        makes sense to use noise-style exploration (add some noise to the output and sample from that: always sample, there is no random or maybe uniform sampling, but that's probably stupid)
        - what about epsilon for continuous actions? if epsilon: uniform sample? else: normal sample according to the nn-generated distribution.
"""
