# Copyright 2018 The Rlgraph Authors, All Rights Reserved.
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

from rlgraph.components.policies.policy import Policy
from rlgraph.utils.decorators import rlgraph_api


class BaselinePolicy(Policy):


    @rlgraph_api
    def get_state_values_logits_probabilities_log_probs(self, nn_input, internal_states=None):
        nn_output = self.get_nn_output(nn_input, internal_states)
        out = self.action_adapter.get_logits_probabilities_log_probs(nn_output["output"])

        state_values = out["state_values"]
        logits = out["logits"]
        probs = out["probabilities"]
        log_probs = out["log_probs"]

        if self.batch_apply is True:
            state_values = self.unfolder.apply(state_values, nn_input)
            logits = self.unfolder.apply(logits, nn_input)
            probs = self.unfolder.apply(probs, nn_input)
            log_probs = self.unfolder.apply(log_probs, nn_input)

        return dict(state_values=state_values, logits=logits, probabilities=probs, log_probs=log_probs,
                    last_internal_states=nn_output.get("last_internal_states"))
