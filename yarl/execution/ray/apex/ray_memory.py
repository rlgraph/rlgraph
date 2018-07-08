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

from yarl.backend_system import get_distributed_backend

if get_distributed_backend() == "ray":
    import ray


@ray.remote
class RayMemory(object):
    """
    An in-memory prioritized replay worker used to aaccelerate memory interaction in Ape-X.
    """
    def __init__(self, memory_spec):
        # TODO create in memory memory
        self.memory = None

    def get_batch(self):
        """
        Samples a batch from the replay memory.

        Returns:
            dict, ndarray: Sample batch and indices sampled.

        """
        pass

    def observe(self, states, actions, internals, rewards, terminals):
        """
        Observes experience(s), see agent observe api for more.

        """
        pass

    def update_priorities(self, indices, loss):
        """
        Updates priorities of provided indices in replay memory via externally
        provided loss.

        Args:
            indices (ndarray): Indices to update in replay memory.
            loss (ndarray):  Loss values for indices.
        """
        pass