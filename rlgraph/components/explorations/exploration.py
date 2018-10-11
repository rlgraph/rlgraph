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
from rlgraph.utils.rlgraph_errors import RLGraphError
from rlgraph.components.component import Component
from rlgraph.components.explorations.epsilon_exploration import EpsilonExploration
from rlgraph.components.common.noise_components import NoiseComponent
from rlgraph.spaces import IntBox, FloatBox
from rlgraph.spaces.space_utils import sanity_check_space
from rlgraph.utils.util import dtype
from rlgraph.utils.decorators import rlgraph_api, graph_fn

if get_backend() == "tf":
    import tensorflow as tf
elif get_backend() == "pytorch":
    import torch


class Exploration(Component):
    """
    A Component that can be plugged on top of a Policy's output to produce action choices.
    It includes noise and/or epsilon-based exploration options as well as an out-Socket to draw actions from
    the Policy's distribution - either by sampling or by deterministically choosing the max-likelihood value.
    """
    def __init__(self, epsilon_spec=None, noise_spec=None, scope="exploration", **kwargs):
        """
        Args:
            epsilon_spec (any): The spec or Component object itself to construct an EpsilonExploration Component.
            noise_spec (dict): The specification dict for a noise generator that adds noise to the NN's output.
        """
        super(Exploration, self).__init__(scope=scope, **kwargs)

        self.action_space = None  # The actual action space (may not have batch-rank, just the plain space)

        self.epsilon_exploration = None
        self.noise_component = None

        # For define-by-run sampling.
        self.sample_obj = None

        # Don't allow both epsilon and noise component
        if epsilon_spec and noise_spec:
            raise RLGraphError("Cannot use both epsilon exploration and a noise component at the same time.")

        # Add epsilon component.
        if epsilon_spec:
            self.epsilon_exploration = EpsilonExploration.from_spec(epsilon_spec)
            self.add_components(self.epsilon_exploration)

            # Define our interface.
            @rlgraph_api(component=self)
            def get_action(self, actions, time_step, use_exploration=True):
                """
                Action depends on time-step (e.g. epsilon-decay).
                """
                epsilon_decisions = self.epsilon_exploration.do_explore(actions, time_step)
                return self._graph_fn_pick(use_exploration, epsilon_decisions, actions)

        # Add noise component.
        elif noise_spec:
            self.noise_component = NoiseComponent.from_spec(noise_spec)
            self.add_components(self.noise_component)

            @rlgraph_api(component=self)
            def get_action(self, actions, time_step=0, use_exploration=True):
                """
                Noise is added to the sampled action.
                """
                noise = self.noise_component.get_noise()
                return self._graph_fn_add_noise(use_exploration, noise, actions)

        # Don't explore at all. Simple pass-through.
        else:
            @rlgraph_api(component=self)
            def get_action(self, actions, time_step=0, use_exploration=False):
                """
                Action is returned as is.
                """
                return actions

    def check_input_spaces(self, input_spaces, action_space=None):
        action_sample_space = input_spaces["actions"]

        if get_backend() == "tf":
            sanity_check_space(action_sample_space, must_have_batch_rank=True)

        assert action_space is not None
        self.action_space = action_space

        if self.epsilon_exploration and self.noise_component:
            # Check again at graph creation? This is currently redundant to the check in __init__
            raise RLGraphError("Cannot use both epsilon exploration and a noise component at the same time.")

        if self.epsilon_exploration:
            sanity_check_space(self.action_space, allowed_types=[IntBox], must_have_categories=True,
                               num_categories=(1, None))
        elif self.noise_component:
            sanity_check_space(self.action_space, allowed_types=[FloatBox])

    @graph_fn
    def _graph_fn_pick(self, use_exploration, epsilon_decisions, sample):
        """
        Exploration for discrete action spaces.
        Either pick a random action (if `use_exploration` and `epsilon_decision` are True),
            or return non-exploratory action.

        Args:
            use_exploration (DataOp): The master switch determining, whether to use exploration or not.
            epsilon_decisions (DataOp): The bool coming from the epsilon-exploration component specifying
                whether to use exploration or not (per batch item).
            sample (DataOp): The output from a distribution's "sample_deterministic" OR "sample_stochastic".

        Returns:
            DataOp: The DataOp representing the action. This will match the shape of self.action_space.
        """
        if get_backend() == "tf":
            random_actions = tf.random_uniform(
                shape=tf.shape(sample),
                maxval=self.action_space.num_categories,
                dtype=dtype("int")
            )

            if use_exploration is False:
                return sample
            else:
                return tf.where(
                    # `use_exploration` given as actual bool or as tensor?
                    condition=epsilon_decisions if use_exploration is True else tf.logical_and(
                        use_exploration, epsilon_decisions
                    ),
                    x=random_actions,
                    y=sample
                )
        elif get_backend() == "pytorch":
            # N.b. different order versus TF because we dont want to execute the sampling below.
            if use_exploration is False:
                    return sample

            if self.sample_obj is None:
                # Don't create new sample objects very time.
                self.sample_obj = torch.distributions.Uniform(0, self.action_space.num_categories)

            random_actions = self.sample_obj.sample(sample.shape).int()
            if use_exploration is True:
                return torch.where(epsilon_decisions, random_actions, sample)
            else:
                if not isinstance(use_exploration, torch.ByteTensor):
                    use_exploration = use_exploration.byte()
                if not isinstance(epsilon_decisions, torch.ByteTensor):
                    epsilon_decisions = epsilon_decisions.byte()
                return torch.where(use_exploration & epsilon_decisions, random_actions, sample)

    @graph_fn
    def _graph_fn_add_noise(self, use_exploration, noise, sample):
        """
        Noise for continuous action spaces.
        Return the action with added noise.

        Args:
            use_exploration (DataOp): The master switch determining, whether to add noise or not.
            noise (DataOp): The noise coming from the noise component.
            sample (DataOp): The output from a distribution's "sample_deterministic" or "sample_stochastic" API-method.

        Returns:
            DataOp: The DataOp representing the action. This will match the shape of self.action_space.

        """
        if get_backend() == "tf":
            return tf.cond(
                use_exploration, true_fn=lambda: sample + noise, false_fn=lambda: sample
            )
        elif get_backend() == "pytorch":
            if use_exploration:
                return sample + noise
            else:
                return sample
