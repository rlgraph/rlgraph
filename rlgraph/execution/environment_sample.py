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


class EnvironmentSample(object):
    """
    Represents a sampled trajectory from an environment.
    """
    def __init__(
        self,
        sample_batch,
        batch_size=None,
        metrics=None,
        **kwargs
    ):
        """
        Args:
            sample_batch (dict): Dict containing sample trajectories.
            **kwargs (dict): Any additional information relevant for processing the sample.
        """
        self.sample_batch = sample_batch
        self.batch_size = batch_size
        self.metrics = metrics
        self.kwargs = kwargs

    def get_batch(self):
        """
        Get experience sample in insert format.

        Returns:
            dict: Sample batch.
        """
        return self.sample_batch

    def get_metrics(self):
        return self.metrics


