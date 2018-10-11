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
from rlgraph.components.component import Component
from rlgraph.components.common.decay_components import DecayComponent
from rlgraph.spaces.space_utils import sanity_check_space
from rlgraph.utils.decorators import rlgraph_api, graph_fn

if get_backend() == "tf":
    import tensorflow as tf
elif get_backend() == "pytorch":
    import torch


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

        # The space of the samples that we have to produce epsilon decisions for.
        self.sample_space = None

        # Our (epsilon) Decay-Component.
        self.decay_component = DecayComponent.from_spec(decay_spec)

        # Add the decay component and make time_step our (only) input.
        self.add_components(self.decay_component)

    def check_input_spaces(self, input_spaces, action_space=None):
        # Require at least a batch-rank in the incoming samples.
        self.sample_space = input_spaces["sample"]
        if get_backend() == "tf":
            sanity_check_space(self.sample_space, must_have_batch_rank=True)

    @rlgraph_api
    def do_explore(self, sample, time_step=0):
        """
        API-method taking a timestep and returning a bool type tensor on whether to explore or not (per batch item).

        Args:
            sample (SingleDataOp): A data sample from which we can extract the batch size.
            time_step (SingleDataOp): The current global time step.

        Returns:
            SingleDataOp: Single decisions over a batch on whether to explore or not.
        """
        decayed_value = self.decay_component.decayed_value(time_step)
        return self._graph_fn_get_random_actions(decayed_value, sample)

    @graph_fn
    def _graph_fn_get_random_actions(self, decayed_value, sample):
        if get_backend() == "tf":
            shape = tf.shape(sample)
            batch_time_shape = (shape[0],) + ((shape[1],) if self.sample_space.has_time_rank is True else ())
            return tf.random_uniform(shape=batch_time_shape) < decayed_value
        elif get_backend() == "pytorch":
            if sample.dim() == 0:
                sample = sample.unsqueeze(-1)
            shape = sample.shape
            batch_time_shape = (shape[0],) + ((shape[1],) if self.sample_space.has_time_rank is True else ())
            x = torch.rand(batch_time_shape) < decayed_value
            return x
