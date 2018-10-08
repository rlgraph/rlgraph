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

from rlgraph.components.component import Component
from rlgraph.components.neural_networks.policy import Policy
from rlgraph.components.helpers import dynamic_batching


# Wrap in dynamic batching module.
@dynamic_batching.batch_fn
def get_state_values_logits_probabilities_log_probs(self_, nn_input_, internal_states_):
    return self_.policy.get_state_values_logits_probabilities_log_probs(nn_input_, internal_states_, return_ops=True)


class DynamicBatchingPolicy(Component):
    """
    A dynamic batching policy wraps a policy with DeepMind's custom
    dynamic batching ops which are provided as part of their IMPALA open source
    implementation.
    """
    def __init__(self, policy_spec, minimum_batch_size=1, maximum_batch_size=1024, timeout_ms=100,
                 scope="dynamic-batching-policy", **kwargs):
        """
        Args:
            policy_spec (Union[Optimizer,dict]): A spec dict to construct the Policy that is wrraped by this
                DynamicBatchingPolicy or a Policy object directly.
            minimum_batch_size (int): The minimum batch size to use. Default: 1.
            maximum_batch_size (int): The maximum batch size to use. Default: 1024
            timeout_ms (int): The time out in ms to use when waiting for items on the queue.
                Default: 100ms.
        """
        super(DynamicBatchingPolicy, self).__init__(
            # 3=states, logits, internal_states
            graph_fn_num_outputs=dict(_graph_fn_get_state_values_logits_probabilities_log_probs=5), scope=scope, **kwargs
        )

        # The wrapped, backend-specific policy object.
        self.policy = Policy.from_spec(policy_spec)

        # hack: link in case parent components call APIs of the distribution directly
        self.action_adapter = self.policy.action_adapter
        self.distribution = self.policy.distribution
        self.max_likelihood = True

        # Dynamic batching options.
        self.minimum_batch_size = minimum_batch_size
        self.maximum_batch_size = maximum_batch_size
        self.timeout_ms = timeout_ms

        self.add_components(self.policy)

        # TODO: for now, only define this one API-method as this is the only one used in IMPALA.
        # TODO: Generalize this component so it can wrap arbitrary other components and simulate their API.
        self.define_api_method("get_state_values_logits_probabilities_log_probs",
                               self._graph_fn_get_state_values_logits_probabilities_log_probs)

    def _graph_fn_get_state_values_logits_probabilities_log_probs(self, nn_input, internal_states=None):
        out = get_state_values_logits_probabilities_log_probs(self, nn_input, internal_states)
        return out

    #def _graph_fn_get_logits_probabilities_log_probs(self, nn_input, internal_states=None):
    #    # Wrap in dynamic batching module.
    #    @dynamic_batching.batch_fn_with_options(minimum_batch_size=self.minimum_batch_size,
    #                                            maximum_batch_size=self.maximum_batch_size,
    #                                            timeout_ms=self.timeout_ms)
    #    def get_logits_probabilities_log_probs(nn_input_, internal_states_):
    #        # TODO potentially assign device
    #        ret = self.policy.get_logits_probabilities_log_probs(nn_input_, internal_states_, return_ops=True)
    #        return ret
    #    out = get_logits_probabilities_log_probs(nn_input, internal_states)
    #    return out
