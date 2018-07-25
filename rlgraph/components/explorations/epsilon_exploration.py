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
from rlgraph.components import Component, DecayComponent, PolynomialDecay, Bernoulli
from rlgraph.components.distributions import Bernoulli

if get_backend() == "tf":
    import tensorflow as tf


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
    def __init__(self, decay_spec=None, scope="epsilon-exploration", **kwargs):
        """
        Keyword Args:
            decay_spec (Optional[dict,DecayComponent]): The spec-dict for the DecayComponent to use or a DecayComponent
                object directly.

        Keyword Args:
            Used as decay_spec (only if `decay_spec` not given) to construct the DecayComponent.
        """
        super(EpsilonExploration, self).__init__(scope=scope, **kwargs)

        # Our (epsilon) Decay-Component.
        self.decay_component = DecayComponent.from_spec(decay_spec)
        # Our Bernoulli distribution to figure out whether we should explore or not.
        self.bernoulli_component = Bernoulli()

        # Add the decay component and make time_step our (only) input.
        self.add_components(self.decay_component, self.bernoulli_component)

    def do_explore(self, time_step):
        """
        API-method taking a timestep and returning a bool on whether to explore or not.

        Args:
            time_step (SingleDataOp): The current global time step.

        Returns:
            bool: Whether to explore or not.
        """
        decayed_value = self.call(self.decay_component.decayed_value, time_step)
        return self.call(self.bernoulli_component.sample_stochastic, decayed_value)


