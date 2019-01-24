# Copyright 2018/2019 The RLgraph authors. All Rights Reserved.
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

import copy

from rlgraph import get_backend
from rlgraph.utils.decorators import rlgraph_api
from rlgraph.components.component import Component
from rlgraph.components.layers.preprocessing import ReShape
from rlgraph.utils.util import force_tuple, force_list


class Stack(Component):
    """
    A component container stack that incorporates one or more sub-components some of whose API-methods
    (default: only `apply`) are automatically connected with each other (in the sequence the sub-Components are given
    in the c'tor), resulting in an API of the Stack.
    All sub-components' API-methods need to match in the number of input and output values. E.g. the third
    sub-component's api-metehod's number of return values has to match the forth sub-component's api-method's number of
    input parameters.
    """
    def __init__(self, *sub_components, **kwargs):
        """
        Args:
            sub_components (Union[Component,List[Component]]): The sub-components to add to the Stack and connect
                to each other.

        Keyword Args:
            api_methods (List[Union[str,Tuple[str,str],dict]]): A list of strings of API-methods names to connect
                through the stack.
                Defaults to {"apply"}. All sub-Components must implement all API-methods in this set.
                Alternatively, this set may contain tuples (1st item is the final Stack's API method name, 2nd item
                is the name of the API-methods of the sub-Components to connect through).
                E.g. api_methods={("stack_run", "run")}. This will create "stack_run" for the Stack, which will call
                - one by one - all the "run" methods of the sub-Components.
                Alternatively, this set may contain spec-dicts with keys:
                `api` (exposed final API-method name), `component_api` (sub-Components API-method names to connect
                through), `function` (the custom API-function to use), `fold_time_rank` (whether to fold a time
                rank into a batch rank at the beginning), `unfold_time_rank` (whether to unfold the time rank
                at the end).

                Connecting always works by first calling the first sub-Component's API-method, then - with the
                result - calling the second sub-Component's API-method, etc..
                This is done for all API-methods in the given set, plus - optionally - time rank folding and unfolding
                at the beginning and/or end.
        """
        self.api_methods_options = kwargs.pop("api_methods", ["apply"])
        super(Stack, self).__init__(*sub_components, scope=kwargs.pop("scope", "stack"), **kwargs)

        self.num_allowed_inputs = None
        self.num_allowed_returns = None

        self.map_api_to_sub_components_api = dict()

        self.folder = ReShape(fold_time_rank=True, scope="time-rank-folder_")
        self.unfolder = ReShape(unfold_time_rank=True, scope="time-rank-unfolder_")

        self.add_components(self.folder, self.unfolder)

        self._build_stack(self.api_methods_options)

    def _build_stack(self, api_methods):
        """
        For each api-method in set `api_methods`, automatically create this Stack's own API-method by connecting
        through all sub-Component's API-methods. This is skipped if this Stack already has a custom API-method
        by that name.

        Args:
            api_methods (List[Union[str,Tuple[str,str],dict]]): See ctor kwargs.
        """
        # Loop through the API-method set and register each one.
        for api_method_spec in api_methods:
            function_to_use = None
            fold_time_rank = False
            unfold_time_rank = False

            # Detailed spec-dict given.
            if isinstance(api_method_spec, dict):
                stack_api_method_name = api_method_spec["api"]
                component_api_method_name = api_method_spec.get("component_api", stack_api_method_name)
                function_to_use = api_method_spec.get("function")
                fold_time_rank = api_method_spec.get("fold_time_rank", False)
                unfold_time_rank = api_method_spec.get("unfold_time_rank", False)
            # API-method of sub-Components and this Stack should have different names.
            elif isinstance(api_method_spec, tuple):
                # Custom method given, use that instead of creating one automatically.
                if callable(api_method_spec[1]):
                    stack_api_method_name = component_api_method_name = api_method_spec[0]
                    function_to_use = api_method_spec[1]
                else:
                    stack_api_method_name, component_api_method_name = api_method_spec[0], api_method_spec[1]
            # API-method of sub-Components and this Stack should have the same name.
            else:
                stack_api_method_name = component_api_method_name = api_method_spec

            self.map_api_to_sub_components_api[stack_api_method_name] = component_api_method_name

            # API-method for this Stack does not exist yet -> Automatically create it.
            if not hasattr(self, stack_api_method_name):
                # Custom API-method is given (w/o decorator) -> Call the decorator directly here to register it.
                if function_to_use is not None:
                    rlgraph_api(api_method=function_to_use, component=self, name=stack_api_method_name)
                # No API-method given -> Create auto-API-method and set it up through decorator.
                else:
                    self.build_auto_api_method(
                        stack_api_method_name, component_api_method_name, fold_time_rank=fold_time_rank,
                        unfold_time_rank=unfold_time_rank
                    )

    def build_auto_api_method(self, stack_api_method_name, sub_components_api_method_name,
                              fold_time_rank=False, unfold_time_rank=False, ok_to_overwrite=False):
        """
        Creates and registers an auto-API method for this stack.

        Args:
            stack_api_method_name (str): The name for the (exposed) API-method of the Stack.

            sub_components_api_method_name (str): The name of the single sub-components' API-methods to call one after
                another.

            ok_to_overwrite (Optional[bool]): Set to True if we know we are overwriting
        """
        @rlgraph_api(name=stack_api_method_name, component=self, ok_to_overwrite=ok_to_overwrite)
        def method(self_, *inputs, **kwargs):
            # Fold time rank? For now only support 1st arg folding/unfolding.
            original_input = inputs[0]
            if fold_time_rank is True:
                args_ = tuple([self.folder.apply(original_input)] + list(inputs[1:]))
            else:
                # TODO: If only unfolding: Assume for now that 2nd input is the original one (so we can infer
                # TODO: batch/time dims).
                if unfold_time_rank is True:
                    assert len(inputs) >= 2, \
                        "ERROR: In Stack: If unfolding w/o folding, second arg must be the original input!"
                    original_input = inputs[1]
                    args_ = tuple([inputs[0]] + list(inputs[2:]))
                else:
                    args_ = inputs
            kwargs_ = kwargs

            for i, sub_component in enumerate(self_.sub_components.values()):  # type: Component
                if sub_component.scope in ["time-rank-folder_", "time-rank-unfolder_"]:
                    continue
                # TODO: python-Components: For now, we call each preprocessor's graph_fn
                #  directly (assuming that inputs are not ContainerSpaces).
                if self_.backend == "python" or get_backend() == "python":
                    graph_fn = getattr(sub_component, "_graph_fn_" + sub_components_api_method_name)
                    # if sub_component.api_methods[components_api_method_name].add_auto_key_as_first_param:
                    #    results = graph_fn("", *args_)  # TODO: kwargs??
                    # else:
                    results = graph_fn(*args_)
                elif get_backend() == "pytorch":
                    # Do NOT convert to tuple, has to be in unpacked again immediately.n
                    results = getattr(sub_component, sub_components_api_method_name)(*force_list(args_))
                else:  # if get_backend() == "tf":
                    results = getattr(sub_component, sub_components_api_method_name)(*args_, **kwargs_)

                # Recycle args_, kwargs_ for reuse in next sub-Component's API-method call.
                if isinstance(results, dict):
                    args_ = ()
                    kwargs_ = results
                else:
                    args_ = force_tuple(results)
                    kwargs_ = {}

            if args_ == ():
                # Unfold time rank? For now only support 1st arg folding/unfolding.
                if unfold_time_rank is True:
                    assert len(kwargs_) == 1,\
                        "ERROR: time-rank-unfolding not supported for more than one NN-return value!"
                    key = next(iter(kwargs_))
                    kwargs_ = {key: self.unfolder.apply(kwargs_[key], original_input)}
                return kwargs_
            else:
                # Unfold time rank? For now only support 1st arg folding/unfolding.
                if unfold_time_rank is True:
                    assert len(args_) == 1,\
                        "ERROR: time-rank-unfolding not supported for more than one NN-return value!"
                    args_ = tuple([self.unfolder.apply(args_[0], original_input)] +
                                  list(args_[1 if fold_time_rank is True else 2:]))
                if len(args_) == 1:
                    return args_[0]
                else:
                    return args_

    @classmethod
    def from_spec(cls, spec=None, **kwargs):
        spec_deepcopy = copy.deepcopy(spec)
        if isinstance(spec, dict):
            kwargs["_args"] = list(spec_deepcopy.pop("layers", []))
        elif isinstance(spec, (tuple, list)):
            kwargs["_args"] = spec_deepcopy
            spec_deepcopy = None
        return super(Stack, cls).from_spec(spec_deepcopy, **kwargs)
