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

from yarl.components import Component


class FixedLoop(Component):
    """
    A FixedLoop component is used to iteratively call graph functions of
    another computation, e.g. in an optimization.
    """

    def __init__(self, iterations, call_component, scope="fixed-loop", **kwargs):
        """
        Args:
            iterations (int): How often the
            call_component (Component): Component providing graph fn to call within loop.
        """
        assert iterations > 0
        super(FixedLoop, self).__init__(scope=scope, **kwargs)
        self.iterations = iterations
        self.call_component = call_component
        self.define_outputs("fixed_loop_result")

    def _graph_fn_call_loop(self):
        pass
