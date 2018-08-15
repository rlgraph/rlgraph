# Copyright 2018 The RLgraph authors, All Rights Reserved.
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

from rlgraph.components.component import Component
from rlgraph.components.neural_networks.preprocessor_stack import PreprocessorStack
from rlgraph.components.neural_networks.policy import Policy
from rlgraph.components.explorations.exploration import Exploration


class ActorComponent(Component):
    """
    A Component that incorporates an entire pipeline from env state to an action choice.
    Includes preprocessor, policy and exploration sub-components.

    API:
        get_preprocessed_state_and_action(state, time_step, use_exploration) ->
    """
    def __init__(self, preprocessor_spec, policy_spec, exploration_spec, max_likelihood=True,
                 **kwargs):
        """
        Args:
            preprocessor_spec ():
            policy_spec ():
            exploration_spec ():
            max_likelihood (bool): Whether to pick the max-likelihood action in case of non-exploratory actions.
                Will send max-likelihood pick (from the policy's distribution)
        """
        super(ActorComponent, self).__init__(scope=kwargs.pop("scope", "actor-component"), **kwargs)

        self.preprocessor = PreprocessorStack.from_spec(preprocessor_spec)
        self.policy = Policy.from_spec(policy_spec)
        self.exploration = Exploration.from_spec(exploration_spec)

        self.add_components(self.preprocessor, self.policy, self.exploration)

    # @rlgraph.api_method
    def get_preprocessed_state_and_action(self, states, internal_states=None, time_step=0, use_exploration=True):
        """
        API-method to get an action based on a raw state from an Env along with the preprocessed state.

        Args:
            states (DataOp): The states coming directly from the environment.
            internal_states (DataOp): The initial internal states to use (in case of an RNN network).
            time_step (DataOp): The current time step.
            use_exploration (Optional[DataOp]): Whether to use exploration or not.

        Returns:
            tuple:
                - DataOp: The preprocessed states.
                - DataOp: The chosen action.
        """
        preprocessed_states = self.call(self.preprocessor.preprocess, states)
        if self.max_likelihood is True:
            sample = self.call(self.policy.get_max_likelihood_action, preprocessed_states, internal_states)
        else:
            sample = self.call(self.policy.get_stochastic_action, preprocessed_states, internal_states)
        actions = self.call(self.exploration.get_action, sample, time_step, use_exploration)
        return preprocessed_states, actions
