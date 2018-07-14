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

from six.moves import xrange as _range
from yarl.backend_system import get_distributed_backend
from yarl.execution.ray.apex.apex_memory import ApexMemory

if get_distributed_backend() == "ray":
    import ray


@ray.remote
class RayMemoryActor(object):
    """
    An in-memory prioritized replay worker
    used to accelerate memory interaction in Ape-X.
    """
    def __init__(self, memory_spec, batch_size):
        self.memory = ApexMemory.from_spec(memory_spec)
        self.batch_size = batch_size

    def get_batch(self):
        """
        Samples a batch from the replay memory.

        Returns:
            dict, ndarray: Sample batch and indices sampled.

        """
        return self.memory.get_records(self.batch_size)

    def observe(self, records):
        """
        Observes experience(s).

        N.b. For performance reason, data layout is slightly different for apex.
        """
        num_records = len(records['states'])
        for i in _range(num_records):
            self.memory.insert_records((
                records['states'][i],
                records['actions'][i],
                records['rewards'][i],
                records['terminal'][i]
            ))

    def update_priorities(self, indices, loss):
        """
        Updates priorities of provided indices in replay memory via externally
        provided loss.

        Args:
            indices (ndarray): Indices to update in replay memory.
            loss (ndarray):  Loss values for indices.
        """
        self.memory.update_priorities(indices, loss)