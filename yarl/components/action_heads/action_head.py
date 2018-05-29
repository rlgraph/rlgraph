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

import numpy as np

from yarl import YARLError, backend
from yarl.components import Component
from yarl.components.distributions import Categorical, NNOutputCleanup
from yarl.spaces import IntBox
from .epsilon_exploration import EpsilonExploration


class ActionHeadComponent(Component):
    """
    A Component that can be plugged on top of an NN-action output as is to produce action choices.
    It includes noise and epsilon-based exploration options as well as drawing actions from different,
    possible NN-specified distributions - either by sampling or by deterministically choosing the max-likelihood
    value.

    API:
    ins:
        time_step (int): The current global time step (used to determine the extend of the exploration).
        nn_output (any): The NN-output specifying the parameters of an action distribution.
    outs:
        action (any): A single action chose according to our exploration settings and NN-output.
        # TODO: actions (any): A batch of actions taken from a batch of NN-outputs without any exploration.
    """
    def __init__(self, action_space, add_softmax=False, policy="max-likelihood", epsilon_spec=None,
                 noise_spec=None, scope="action-head", **kwargs):
        """
        Args:
            action_space (Space): The action Space. Outputs of the "action" Socket must be members of this Space.
            add_softmax (bool): Whether to softmax the NN outputs (before further processing them).
            policy (str): One of: "max-likelihood", "sample".
            epsilon_spec (any): The spec or Component object itself to construct an EpsilonExploration Component.
            #noise_spec (dict): The specification dict for a noise generator that adds noise to the NN's output.
        """
        super(ActionHeadComponent, self).__init__(scope=scope, **kwargs)

        self.add_softmax = add_softmax

        self.action_space = action_space
        assert self.action_space.has_batch_rank, "ERROR: `action_space` does not have batch rank!"

        # The Distribution to sample (or pick) actions from.
        # Discrete action space -> Categorical distribution.
        if isinstance(self.action_space, IntBox):
            self.nn_cleanup = NNOutputCleanup(self.action_space)
            self.action_distribution = Categorical()
        else:
            raise YARLError("ERROR: Space of out-Socket `action` is of type {} and not allowed in {} Component!".
                            format(type(self.action_space).__name__, self.name))

        self.policy = policy
        self.epsilon_exploration = Component.from_spec(type=EpsilonExploration, spec=epsilon_spec)

        self.define_inputs("time_step", "nn_output")
        self.define_outputs("action")

        # Add NN-cleanup component and connect to our "nn_output" in-Socket.
        self.add_component(self.nn_cleanup, connect="nn_output")

        # Add action-distribution component and connect to the NN-cleanup.
        self.add_component(self.action_distribution,
                           connect=dict(parameters=(self.nn_cleanup, "parameters"),
                                        max_likelihood=True if self.policy == "max-likelihood" else False)
                           )

        # Add epsilon Component and connect accordingly.
        if self.epsilon_exploration is not None:
            self.add_component(self.epsilon_exploration, connect="time_step")

        # Add our own computation and connect its output to the "action" Socket.
        self.add_computation(inputs=[(self.epsilon_exploration, "do_explore"), (self.action_distribution, "draw")],
                             outputs="action",
                             method=self._computation_pick)

    def create_variables(self, input_spaces):
        flat_action_space = input_spaces["nn_output"]
        assert flat_action_space.flat_dim == self.action_space.flat_dim, \
            "ERROR: The flat_dims of incoming NN-output ({}) and our action_space ({}) don't match!". \
            format(flat_action_space.flat_dim, self.action_space.flat_dim)

    def _computation_pick(self, do_explore, action):
        """
        Args:
            do_explore (DataOp): The bool coming from the epsilon-exploration component specifying
                whether to use exploration or not.
            action (DataOp): The output from our action-distribution (parameterized by the neural network) and
                drawn either by sampling or by max-likelihood picking.

        Returns:
            DataOp: The DataOp representing the action. This will match the shape of self.action_space.
        """
        logits = np.ones((1,) + self.action_space.shape)  # add artificial batch rank
        if backend() == "tf":
            import tensorflow as tf
            return tf.cond(do_explore,
                           true_fn=lambda: tf.multinomial(logits=logits, num_samples=1),
                           false_fn=lambda: action)


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
