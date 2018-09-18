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

from rlgraph.components.memories.memory import Memory
from rlgraph.components.memories.fifo_queue import FIFOQueue
from rlgraph.components.memories.prioritized_replay import PrioritizedReplay
from rlgraph.components.memories.replay_memory import ReplayMemory
from rlgraph.components.memories.ring_buffer import RingBuffer
from rlgraph.components.memories.mem_prioritized_replay import MemPrioritizedReplay


Memory.__lookup_classes__ = dict(
    fifo=FIFOQueue,
    fifoqueue=FIFOQueue,
    prioritized=PrioritizedReplay,
    prioritizedreplay=PrioritizedReplay,
    prioritizedreplaybuffer=PrioritizedReplay,
    mem_prioritized_replay=MemPrioritizedReplay,
    replay=ReplayMemory,
    replaybuffer=ReplayMemory,
    replaymemory=ReplayMemory,
    ringbuffer=RingBuffer
)
Memory.__default_constructor__ = ReplayMemory

__all__ = ["Memory"] + \
          list(set(map(lambda x: x.__name__, Memory.__lookup_classes__.values())))

