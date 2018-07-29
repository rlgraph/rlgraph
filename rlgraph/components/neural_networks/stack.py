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

from rlgraph.components import Component
from rlgraph.utils.util import force_tuple


class Stack(Component):
    """
    A component container stack that incorporates one or more sub-components some of whose API-methods
    (default: only `apply`) are automatically connected with each other (in the sequence the sub-Components are given
    in the c'tor), resulting in an API of the Stack.
    All sub-components' API-methods need to match in the number of input and output values. E.g. the third
    sub-component's api-metehod's number of return values has to match the forth sub-component's api-method's number of
    input parameters.

    API:
        apply(input[, input2, ...]?): Sends one (or more, depending on the 1st sub-Component's `apply` method)
            DataOpRecord(s) through the stack and returns one (or more, depending on the last sub-Component's `apply`
            method) DataOpRecords.
        Optional:
            Other API-methods that all sub-Component have in common.
    """
    def __init__(self, *sub_components, **kwargs):
        """
        Args:
            sub_components (Component): The sub-components to add to the Stack and connect to each other.

        Keyword Args:
            api_methods (Optional[Set[str]]): A set of names of API-methods to connect through the stack.
                Defaults to {"apply"}. All sub-Components must implement all API-methods in this set.
                Alternatively, a tuple can be used  (instead of a string), in which case the first tuple-item
                is used as the Stack's API-method name and the second item is the sub-Components' API-method name.
                E.g. api_methods={("stack_run", "run")}. This will create "stack_run" for the Stack, which will call
                one by one the "run" methods of the sub-Components.

                Connecting always works by first calling the first sub-Component's API-method, then - with the
                result - calling the second sub-Component's API-method, etc..
                This is done for all API-methods in the given set.
        """
        api_methods = kwargs.pop("api_methods", {"apply"})

        super(Stack, self).__init__(*sub_components, **kwargs)

        # For each api-method in the given set, create our own API-method connecting
        # all sub-Component's API-method "through".
        for api_method_name in api_methods:
            if isinstance(api_method_name, tuple):
                stack_api_method_name, components_api_method_name = api_method_name[0], api_method_name[1]
            else:
                stack_api_method_name, components_api_method_name = api_method_name, api_method_name

            def method(self_, *inputs):
                result = inputs
                for sub_component in self_.sub_components.values():
                    result = self_.call(getattr(sub_component, components_api_method_name), *force_tuple(result))
                return result

            # Register `method` to this Component using the custom name given in `api_methods`.
            self.define_api_method(stack_api_method_name, method)

    @classmethod
    def from_spec(cls, spec=None, **kwargs):
        if isinstance(spec, dict):
            kwargs["_args"] = list(spec.pop("layers", []))
        elif isinstance(spec, (tuple, list)):
            kwargs["_args"] = spec
            spec = None
        return super(Stack, cls).from_spec(spec, **kwargs)
