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

from .memory import Memory
from .replay_memory import ReplayMemory
from .ring_buffer import RingBuffer
from .prioritized_replay import PrioritizedReplay


Memory.__lookup_classes__ = dict(
    replay=ReplayMemory,
    replay_buffer=ReplayMemory,
    ring_buffer=RingBuffer,
    prioritized_replay=PrioritizedReplay,
    prioritized_replay_buffer=PrioritizedReplay
)

__all__ = ["Memory", "ReplayMemory", "RingBuffer", "PrioritizedReplay"]

