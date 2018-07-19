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


class EnvironmentSample(object):
    """
    Represents a sampled trajectory from an environment.
    """
    def __init__(
        self,
        states,
        actions,
        rewards,
        terminals,
        metrics=None,
        **kwargs
    ):
        """
        Args:
            states (list): List of states in the sample.
            actions (list): List of actions in the sample.
            rewards (list): List of rewards in the sample.
            terminals (list): List of terminals in the sample.
            metrics Optional[(dict)]: Metrics, e.g. on timing.
            **kwargs (dict): Any additional information relevant for processing the sample.
        """
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals
        self.metrics = metrics
        self.kwargs = kwargs

    def get_batch(self):
        """
        Get experience sample in insert format.
        Returns:
            dict: Sample batch.
        """
        return dict(
            states=self.states,
            actions=self.actions,
            rewards=self.rewards,
            terminals=self.terminals,
        )

    def get_batch_size(self):
        return len(self.states)

    def get_metrics(self):
        return self.metrics


