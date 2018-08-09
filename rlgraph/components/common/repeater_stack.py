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

from rlgraph.components.neural_networks.stack import Stack


class RepeaterStack(Stack):
    """
    A repeater is a special Stack that copies one(!) sub-Component n times and calls certain API-method(s) n times.
    n is the number of repeats.

    API:
        apply(input_) -> call's some API-method on the "repeat-unit" (another Component) n times, each time passing the
            result of the previous repeat and then returning the result of the last repeat.
    """
    def __init__(self, sub_component, repeats=2, scope="repeater", **kwargs):
        """
        Args:
            sub_component (Component): The single sub-Component to repeat (and deepcopy) n times.
            repeats (int): The number of times that the `sub_component`'s API-method(s) should be called.
        """
        self.repeats = repeats
        # Deep copy the sub-Component n times (including riginal sub-Component).
        sub_components = [sub_component] + [
            sub_component.copy(scope=sub_component.scope+"-rep"+str(i+1)) for i in range(self.repeats - 1)
        ]
        # Call the Stack ctor to handle sub-component adding API-method creation.
        super(RepeaterStack, self).__init__(*sub_components, scope=scope, **kwargs)
