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


class MetaGraph(object):
    """
    Represents a single RLgraph meta graph object.
    """
    def __init__(self, root_component, api, num_ops, build_status=False):
        self.root_component = root_component
        self.api = api
        self.num_ops = num_ops
        self._built = build_status

    def set_to_built(self):
        assert not self._built, "ERROR: Cannot set graph to built if already built."
        self._built = True

    @property
    def build_status(self):
        return self._built

