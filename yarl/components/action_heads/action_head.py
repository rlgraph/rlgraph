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

from yarl.components import Component


class ActionHead(Component):
    def __init__(self, action_space,
                 epsilon_component=None, epsilon_strategy="greedy", noise_component=None,
                 scope="action-head", **kwargs):
        super(ActionHead, self).__init__(scope=scope, **kwargs)

        self.define_inputs("nn_output", "time_step")
        self.define_outputs("action")
        self.add_computation(["nn_output", "time_step"], "action", self._computation_act)

    def _computation_act(self, nn_output, time_step):
        """
        Args:
            nn_output (DataOp): The output from the neural network (policy, Q-net, etc..).
            time_step (DataOp): The time-step information needed to figure out the amount and nature of exploration.

        Returns:
            DataOp: The DataOp representing the action. This will match the shape of self.action_space.
        """
        raise NotImplementedError

"""        
        ActionHead
    __init__(action-space, epsilon-comp=epsilon-component if None, no epsilon;
        alternatively: noise-component to be added to continuous action output?
        action-pick=[greedy|sample|random])
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
