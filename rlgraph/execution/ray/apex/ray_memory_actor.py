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
from rlgraph import get_distributed_backend
from rlgraph.execution.ray.apex.apex_memory import ApexMemory
from rlgraph.execution.ray.ray_actor import RayActor

if get_distributed_backend() == "ray":
    import ray


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
        self.clip_rewards = apex_replay_spec.get("clip_rewards", True)
        self.sample_batch_size = apex_replay_spec["sample_batch_size"]
        self.memory = ApexMemory(**apex_replay_spec["memory_spec"])

    @classmethod
    def as_remote(cls, num_cpus=None, num_gpus=None):
        return ray.remote(num_cpus=num_cpus, num_gpus=num_gpus)(cls)

    def get_batch(self):
        """
        Samples a batch from the replay memory.

        Returns:
            dict: Sample batch

        """
        if self.memory.size < self.min_sample_memory_size:
            return None
        else:
            batch, indices, weights = self.memory.get_records(self.sample_batch_size)
            # Merge into one dict to only return one future in ray.
            batch["indices"] = indices
            batch["importance_weights"] = weights
            return batch

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
                records["next_states"][i],
                records["importance_weights"][i]
            ))
            # self.memory.insert_records((
            #     records["states"][i],
            #     records["actions"][i],
            #     rewards[i],
            #     records["terminals"][i],
            #     records["importance_weights"][i]
            # ))

    def update_priorities(self, indices, loss):
        """
        Updates priorities of provided indices in replay memory via externally
        provided loss.

        Args:
            indices (ndarray): Indices to update in replay memory.
            loss (ndarray):  Loss values for indices.
        """
        self.memory.update_records(indices, loss)