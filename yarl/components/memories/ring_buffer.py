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

from yarl.components.memories.memory import Memory
import tensorflow as tf


class RingBuffer(Memory):
    """
    Simple ring-buffer to be used for on-policy sampling based on sample count
    or episodes. Fetches most recently added memories.
    """
    def __init__(
        self,
        record_space,
        capacity=1000,
        name="",
        scope="ring-buffer",
        sub_indexes=None,
    ):
        super(RingBuffer, self).__init__(record_space, capacity, name, scope, sub_indexes)

    def create_variables(self):
        super(RingBuffer, self).create_variables()

        # Main buffer index.
        self.index = self.get_variable(name="index", dtype=int, trainable=False, initializer=0)
        # Number of elements present.
        self.size = self.get_variable(name="size", dtype=int, trainable=False, initializer=0)

    def _computation_insert(self, records):
        pass

    def _computation_get_records(self, num_records):
        pass

    def _computation_get_episodes(self, num_records):
        pass
