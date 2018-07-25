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

import numpy as np
from six.moves import xrange as range_
from rlgraph.backend_system import get_distributed_backend
from rlgraph.execution.ray.apex.apex_memory import ApexMemory
from rlgraph.execution.ray.ray_actor import RayActor

if get_distributed_backend() == "ray":
    import ray


@ray.remote
class RayMemoryActor(RayActor):
    """
    An in-memory prioritized replay worker used to accelerate memory interaction in Ape-X.
    """
    def __init__(self, apex_replay_spec):
        """
        Args:
            apex_replay_spec (dict): Specifies behaviour of this replay actor. Must contain key "memory_spec".
        """
        # N.b. The memory spec contains type PrioritizedReplay because that is
        # used for the agent. We hence do not use from_spec but just read the relevant
        # args.
        self.min_sample_memory_size = apex_replay_spec["min_sample_memory_size"]
        memory_spec = apex_replay_spec["memory_spec"]
        self.clip_rewards = apex_replay_spec.get("clip_rewards", True)

        self.memory = ApexMemory(
            capacity=memory_spec["capacity"],
            alpha=memory_spec.get("alpha", 1.0),
            beta=memory_spec.get("beta", 1.0),
            n_step=memory_spec.get("n_step_adjustment", 1)
        )

    def get_batch(self, batch_size):
        """
        Samples a batch from the replay memory.

        Returns:
            dict, ndarray: Sample batch and indices sampled.

        """
        if self.memory.size < self.min_sample_memory_size:
            return None, None
        else:
            batch, indices, weights = self.memory.get_records(batch_size)
            batch["importance_weights"] = weights
            return batch, indices

    def observe(self, env_sample):
        """
        Observes experience(s).

        N.b. For performance reason, data layout is slightly different for apex.
        """
        records = env_sample.get_batch()
        num_records = len(records['states'])

        # TODO port to tf PR behaviour.
        if self.clip_rewards:
            rewards = np.sign(records["rewards"])
        else:
            rewards = records["rewards"]
        for i in range_(num_records):
            self.memory.insert_records((
                records["states"][i],
                records["actions"][i],
                rewards[i],
                records["terminals"][i],
                records["importance_weights"][i]
            ))

    def update_priorities(self, indices, loss):
        """
        Updates priorities of provided indices in replay memory via externally
        provided loss.

        Args:
            indices (ndarray): Indices to update in replay memory.
            loss (ndarray):  Loss values for indices.
        """
        self.memory.update_records(indices, loss)