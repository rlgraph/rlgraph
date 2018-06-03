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
from yarl.components.common import PolynomialDecay
from yarl.components.distributions import Bernoulli


class EpsilonExploration(Component):
    """
    A component to handle epsilon-exploration functionality. It takes the current time step and outputs a bool
    on whether to explore (uniformly random) or not (greedy or sampling).
    The time step is used by a epsilon-decay component to determine the current epsilon value between 1.0
    and 0.0. The result of this decay is the probability, with which we output "True" (meaning: do explore),
    vs "False" (meaning: do not explore).

    API:
    ins:
        time_step (int): The current time step.
    outs:
        do_explore (bool): The decision whether to explore (do_explore=True; pick uniformly randomly) or
            whether to use a sample (or max-likelihood value) from a distribution (do_explore=False).
    """
    def __init__(self, decay_component=None, scope="epsilon-exploration", **kwargs):
        super(EpsilonExploration, self).__init__(scope=scope, **kwargs)

        # Our (epsilon) Decay-Component.
        self.decay_component = decay_component or PolynomialDecay()
        # Our Bernoulli distribution to figure out whether we should explore or not.
        self.bernoulli_component = Bernoulli()

        # Add the decay component and make time_step our (only) input.
        self.add_component(self.decay_component, connections=["time_step"])  # create our own "time_step" in-Socket.

        # Add the Bernoulli Distribution and make do_explore our only output.
        self.add_component(self.bernoulli_component, connections=[("draw", "do_explore"), ("max_likelihood", False)])

        # Connect the two: Feed the epsilon-value from the decay component as prob-parameter into Bernoulli.
        self.connect((self.decay_component, "value"), (self.bernoulli_component, "parameters"))

