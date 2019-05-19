# Copyright 2018/2019 The Rlgraph Authors, All Rights Reserved.
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
import inspect
import re
import time

# from rlgraph.components.common.container_merger import ContainerMerger
from rlgraph.spaces.space_utils import get_space_from_op
from rlgraph.utils import util
from rlgraph.utils.op_records import GraphFnRecord, APIMethodRecord, DataOpRecord, DataOpRecordColumnIntoAPIMethod, \
    DataOpRecordColumnFromAPIMethod, DataOpRecordColumnIntoGraphFn, DataOpRecordColumnFromGraphFn
from rlgraph.utils.ops import TraceContext
from rlgraph.utils.rlgraph_errors import RLGraphError, RLGraphAPICallParamError, RLGraphVariableIncompleteError, \
    RLGraphInputIncompleteError

# Global registries for Component classes' API-methods and graph_fn.
component_api_registry = {}
component_graph_fn_registry = {}


def rlgraph_api(api_method=None, *, component=None, name=None, returns=None,
                flatten_ops=False, split_ops=False, add_auto_key_as_first_param=False,
                must_be_complete=True, ok_to_overwrite=False, requires_variable_completeness=False):
    """
    API-method decorator used to tag any Component's methods as API-methods.

    Args:
        api_method (callable): The actual function/method to tag as an API method.

        component (Optional[Component]): The Component that the method should belong to. None if `api_method` is
            decorated inside a Component class.

        name (Optional[str]): The name under which the API-method should be registered. This is only necessary if
            the API-method is automatically generated as a thin-wrapper around a graph_fn.

        returns (Optional[int]): If the function is a graph_fn, we may specify, how many return values
            it returns. If None, will try to get this number from looking at the source code or from the Component's
            `num_graph_fn_return_values` property.

        flatten_ops (Union[bool,Set[str]]): Whether to flatten all or some DataOps by creating
            a FlattenedDataOp (with automatic key names).
            Can also be a set of in-Socket names to flatten explicitly (True for all).
            (default: True).

        split_ops (Union[bool,Set[str]]): Whether to split all or some of the already flattened DataOps
            and send the SingleDataOps one by one through the graph_fn.
            Example: Spaces=A=Dict (container), B=int (primitive)
            The graph_fn should then expect for each primitive Space in A:
            _graph_fn(primitive-in-A (Space), B (int))
            NOTE that B will be the same in all calls for all primitive-in-A's.
            (default: True).

        add_auto_key_as_first_param (bool): If `split_ops` is not False, whether to send the
            automatically generated flat key as the very first parameter into each call of the graph_fn.
            Example: Spaces=A=float (primitive), B=Tuple (container)
            The graph_fn should then expect for each primitive Space in B:
            _graph_fn(key, A (float), primitive-in-B (Space))
            NOTE that A will be the same in all calls for all primitive-in-B's.
            The key can now be used to index into variables equally structured as B.
            Has no effect if `split_ops` is False.
            (default: False).

        must_be_complete (bool): Whether the exposed API methods must be input-complete or not.

        ok_to_overwrite (bool): Set to True to indicate that this API-decorator will overwrite an already existing
            API-method in the Component. Default: False.

        requires_variable_completeness (bool): Whether the underlying graph_fn should only be called
            after the Component is variable-complete. By default, only input-completeness is required.

    Returns:
        callable: The decorator function.
    """
    _sanity_check_decorator_options(flatten_ops, split_ops, add_auto_key_as_first_param)

    def decorator_func(wrapped_func):

        def api_method_wrapper(self, *args, **kwargs):
            api_fn_name = name or re.sub(r'^_graph_fn_', "", wrapped_func.__name__)
            # Direct evaluation of function.
            if self.execution_mode == "define_by_run":
                type(self).call_count += 1

                start = time.perf_counter()
                # Check with owner if extra args needed.
                if api_fn_name in self.api_methods and self.api_methods[api_fn_name].add_auto_key_as_first_param:
                    output = wrapped_func(self, "", *args, **kwargs)
                else:
                    output = wrapped_func(self, *args, **kwargs)

                # Store runtime for this method.
                type(self).call_times.append(  # Component.call_times
                    (self.name, wrapped_func.__name__, time.perf_counter() - start)
                )
                return output

            api_method_rec = self.api_methods[api_fn_name]

            # TODO: Remove this code or make smarter: passing dicts is actually ok now.
            # Sanity check input args for accidential dict-return values being passed into the next API as
            # supposed DataOpRecord.
            #dict_args = [next(iter(a.values())) for a in args if isinstance(a, dict)]
            #if len(dict_args) > 0 and isinstance(dict_args[0], DataOpRecord):
            #    raise RLGraphError(
            #        "One of your input args to API-method '{}.{}()' is a dict of DataOpRecords! This is probably "
            #        "coming from a previous call to an API-method (returning a dict) and the DataOpRecord should be "
            #        "extracted by string-key and passed into '{}' "
            #        "directly.".format(api_method_rec.component.global_scope, api_fn_name, api_fn_name)
            #    )
            # Create op-record column to call API method with. Ignore None input params. These should not be sent
            # to the API-method.
            in_op_column = DataOpRecordColumnIntoAPIMethod(
                component=self, api_method_rec=api_method_rec, args=args, kwargs=kwargs
            )
            # Add the column to the API-method record.
            api_method_rec.in_op_columns.append(in_op_column)

            # Check minimum number of passed args.
            minimum_num_call_params = len(in_op_column.api_method_rec.non_args_kwargs) - \
                len(in_op_column.api_method_rec.default_args)
            if len(in_op_column.op_records) < minimum_num_call_params:
                raise RLGraphAPICallParamError(
                    "Number of call params ({}) for call to API-method '{}' is too low. Needs to be at least {} "
                    "params!".format(len(in_op_column.op_records), api_method_rec.name, minimum_num_call_params)
                )

            # Link from incoming op_recs into the new column or populate new column with ops/Spaces (this happens
            # if this call was made from within a graph_fn such that ops and Spaces are already known).
            all_args = [(i, a) for i, a in enumerate(args) if a is not None] + \
                       [(k, v) for k, v in sorted(kwargs.items()) if v is not None]
            flex = None
            build_when_done = False
            for i, (key, value) in enumerate(all_args):
                # Named arg/kwarg -> get input_name from that and peel op_rec.
                if isinstance(key, str):
                    param_name = key
                # Positional arg -> get input_name from input_names list.
                else:
                    slot = key if flex is None else flex
                    if slot >= len(api_method_rec.input_names):
                        raise RLGraphAPICallParamError(
                            "Too many input args given in call to AI-method '{}'! Expected={}, you passed in "
                            "more than {}.".format(api_method_rec.name, len(api_method_rec.input_names), slot)
                        )
                    param_name = api_method_rec.input_names[slot]

                # Var-positional arg, attach the actual position to input_name string.
                if self.api_method_inputs.get(param_name, "") == "*flex":
                    if flex is None:
                        flex = i
                    param_name += "[{}]".format(i - flex)
                # Actual kwarg (not in list of api_method_inputs).
                elif api_method_rec.kwargs_name is not None and param_name not in self.api_method_inputs:
                    param_name = api_method_rec.kwargs_name + "[{}]".format(param_name)

                # We are already in building phase (params may be coming from inside graph_fn).
                if self.graph_builder is not None and self.graph_builder.phase == "building":
                    # If Space not stored yet, determine it from op.
                    assert in_op_column.op_records[i].op is not None
                    if in_op_column.op_records[i].space is None:
                        in_op_column.op_records[i].space = get_space_from_op(in_op_column.op_records[i].op)
                    self.api_method_inputs[param_name] = in_op_column.op_records[i].space
                    # Check input-completeness of Component (but not strict as we are only calling API, not a graph_fn).
                    if self.input_complete is False:
                        # Build right after this loop in case more Space information comes in through next args/kwargs.
                        build_when_done = True

                # A DataOpRecord from the meta-graph.
                elif isinstance(value, DataOpRecord):
                    # Create entry with unknown Space if it doesn't exist yet.
                    if param_name not in self.api_method_inputs:
                        self.api_method_inputs[param_name] = None

                # Fixed value (instead of op-record): Store the fixed value directly in the op.
                else:
                    if self.api_method_inputs.get(param_name) is None:
                        self.api_method_inputs[param_name] = in_op_column.op_records[i].space

            if build_when_done:
                # Check Spaces and create variables.
                self.graph_builder.build_component_when_input_complete(self)

            # Regular API-method: Call it here.
            api_fn_args, api_fn_kwargs = in_op_column.get_args_and_kwargs()

            if api_method_rec.is_graph_fn_wrapper is False:
                return_values = wrapped_func(self, *api_fn_args, **api_fn_kwargs)
            # Wrapped graph_fn: Call it through yet another wrapper.
            else:
                return_values = graph_fn_wrapper(
                    self, wrapped_func, returns, dict(
                        flatten_ops=flatten_ops, split_ops=split_ops,
                        add_auto_key_as_first_param=add_auto_key_as_first_param,
                        requires_variable_completeness=requires_variable_completeness
                    ), *api_fn_args, **api_fn_kwargs
                )

            # Process the results (push into a column).
            out_op_column = DataOpRecordColumnFromAPIMethod(
                component=self,
                api_method_name=api_fn_name,
                args=util.force_tuple(return_values) if type(return_values) != dict else None,
                kwargs=return_values if type(return_values) == dict else None
            )

            # If we already have actual op(s) and Space(s), push them already into the
            # DataOpRecordColumnFromAPIMethod's records.
            if self.graph_builder is not None and self.graph_builder.phase == "building":
                # Link the returned ops to that new out-column.
                for i, rec in enumerate(out_op_column.op_records):
                    out_op_column.op_records[i].op = rec.op
                    out_op_column.op_records[i].space = rec.space
            # And append the new out-column to the api-method-rec.
            api_method_rec.out_op_columns.append(out_op_column)

            # Do we need to return the raw ops or the op-recs?
            # Only need to check if False, otherwise, we return ops directly anyway.
            return_ops = False
            stack = inspect.stack()
            f_locals = stack[1][0].f_locals
            # We may be in a list comprehension, try next frame.
            if f_locals.get(".0"):
                f_locals = stack[2][0].f_locals
            # Check whether the caller component is a parent of this one.
            caller_component = f_locals.get("root", f_locals.get("self_", f_locals.get("self")))

            # Potential call from a lambda.
            if caller_component is None and "fn" in stack[2][0].f_locals:
                # This is the component.
                prev_caller_component = TraceContext.PREV_CALLER
                lambda_obj = stack[2][0].f_locals["fn"]
                if "lambda" in inspect.getsource(lambda_obj):
                    # Try to reconstruct caller by using parent of prior caller.
                    caller_component = prev_caller_component.parent_component

            if caller_component is None:
                raise RLGraphError(
                    "API-method '{}' must have as 1st parameter (the component) either `root` or `self`. Other names "
                    "are not allowed!".format(api_method_rec.name)
                )
            # Not directly called by this method itself (auto-helper-component-API-call).
            # AND call is coming from some caller Component, but that component is not this component
            # OR a parent -> Error.
            elif caller_component is not None and \
                    type(caller_component).__name__ != "MetaGraphBuilder" and \
                    caller_component not in [self] + self.get_parents():
                if not (stack[1][3] == "__init__" and re.search(r'op_records\.py$', stack[1][1])):
                    raise RLGraphError(
                        "The component '{}' is not a child (or grand-child) of the caller ({})! Maybe you forgot to "
                        "add it as a sub-component via `add_components()`.".
                        format(self.global_scope, caller_component.global_scope)
                    )

            # Update trace context.
            TraceContext.PREV_CALLER = caller_component

            for stack_item in stack[1:]:  # skip current frame
                # If we hit an API-method call -> return op-recs.
                if stack_item[3] == "api_method_wrapper" and re.search(r'decorators\.py$', stack_item[1]):
                    break
                # If we hit a graph_fn call -> return ops.
                elif stack_item[3] == "run_through_graph_fn" and re.search(r'graph_builder\.py$', stack_item[1]):
                    return_ops = True
                    break

            if return_ops is True:
                if type(return_values) == dict:
                    return {key: value.op for key, value in out_op_column.get_args_and_kwargs()[1].items()}
                else:
                    tuple_returns = tuple(map(lambda x: x.op, out_op_column.get_args_and_kwargs()[0]))
                    return tuple_returns[0] if len(tuple_returns) == 1 else tuple_returns
            # Parent caller is non-graph_fn: Return op-recs.
            else:
                if type(return_values) == dict:
                    return return_values
                else:
                    tuple_returns = out_op_column.get_args_and_kwargs()[0]
                    return tuple_returns[0] if len(tuple_returns) == 1 else tuple_returns

        func_type = util.get_method_type(wrapped_func)
        is_graph_fn_wrapper = (func_type == "graph_fn")
        api_fn_name = name or (re.sub(r'^_graph_fn_', "", wrapped_func.__name__) if is_graph_fn_wrapper else
                         wrapped_func.__name__)
        api_method_rec = APIMethodRecord(
            func=wrapped_func, wrapper_func=api_method_wrapper,
            name=api_fn_name,
            must_be_complete=must_be_complete, ok_to_overwrite=ok_to_overwrite,
            is_graph_fn_wrapper=is_graph_fn_wrapper, is_class_method=(component is None),
            flatten_ops=flatten_ops, split_ops=split_ops, add_auto_key_as_first_param=add_auto_key_as_first_param,
            requires_variable_completeness=requires_variable_completeness
        )

        # Registers the given method with the Component (if not already done so).
        if component is not None:
            define_api_method(component, api_method_rec, copy_record=False)
        # Registers the given function with the Component sub-class so we can define it for each
        # constructed instance of that sub-class.
        else:
            cls = wrapped_func.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0]
            if cls not in component_api_registry:
                component_api_registry[cls] = []
            component_api_registry[cls].append(api_method_rec)

        return api_method_wrapper

    if api_method is None:
        return decorator_func
    else:
        return decorator_func(api_method)


def graph_fn(graph_fn=None, *, component=None, returns=None,
             flatten_ops=False, split_ops=False, add_auto_key_as_first_param=False,
             requires_variable_completeness=False):
    """
    Graph_fn decorator used to tag any Component's graph_fn (that is not directly wrapped by an API-method) as such.

    Args:
        graph_fn (callable): The actual graph_fn to tag.

        component (Optional[Component]): The Component that the graph function should belong to. None if `graph_fn` is
            decorated inside a Component class.

        returns (Optional[int]): How many return values it returns. If None, will try to get this number from looking at the source code or from the Component's
            `num_graph_fn_return_values` property.

        flatten_ops (Union[bool,Set[str]]): Whether to flatten all or some DataOps by creating
            a FlattenedDataOp (with automatic key names).
            Can also be a set of in-Socket names to flatten explicitly (True for all).
            Default: True.

        split_ops (Union[bool,Set[str]]): Whether to split all or some of the already flattened DataOps
            and send the SingleDataOps one by one through the graph_fn.
            Example: Spaces=A=Dict (container), B=int (primitive)
            The graph_fn should then expect for each primitive Space in A:
            _graph_fn(primitive-in-A (Space), B (int))
            NOTE that B will be the same in all calls for all primitive-in-A's.
            Default: True.

        add_auto_key_as_first_param (bool): If `split_ops` is not False, whether to send the
            automatically generated flat key as the very first parameter into each call of the graph_fn.
            Example: Spaces=A=float (primitive), B=Tuple (container)
            The graph_fn should then expect for each primitive Space in B:
            _graph_fn(key, A (float), primitive-in-B (Space))
            NOTE that A will be the same in all calls for all primitive-in-B's.
            The key can now be used to index into variables equally structured as B.
            Has no effect if `split_ops` is False.
            Default: False.

        requires_variable_completeness (bool): Whether the underlying graph_fn should only be called
            after the Component is variable-complete. By default, only input-completeness is required.

    Returns:
        callable: The decorator function.
    """
    _sanity_check_decorator_options(flatten_ops, split_ops, add_auto_key_as_first_param)

    def decorator_func(wrapped_func):
        def _graph_fn_wrapper(self, *args, **kwargs):
            if self.execution_mode == "define_by_run":
                # Direct execution.
                return self.graph_builder.execute_define_by_run_graph_fn(self, wrapped_func,  dict(
                        flatten_ops=flatten_ops, split_ops=split_ops,
                        add_auto_key_as_first_param=add_auto_key_as_first_param
                        ), *args, **kwargs)
            else:
                # Wrap construction of graph functions with op records.
                return graph_fn_wrapper(
                    self, wrapped_func, returns, dict(
                        flatten_ops=flatten_ops, split_ops=split_ops,
                        add_auto_key_as_first_param=add_auto_key_as_first_param,
                        requires_variable_completeness=requires_variable_completeness
                    ), *args, **kwargs
                )

        graph_fn_rec = GraphFnRecord(
            func=wrapped_func, wrapper_func=_graph_fn_wrapper, is_class_method=(component is None),
            flatten_ops=flatten_ops, split_ops=split_ops,
            add_auto_key_as_first_param=add_auto_key_as_first_param,
            requires_variable_completeness=requires_variable_completeness
        )

        # Registers the given method with the Component (if not already done so).
        if component is not None:
            define_graph_fn(component, graph_fn_rec, copy_record=False)
        # Registers the given function with the Component sub-class so we can define it for each
        # constructed instance of that sub-class.
        else:
            cls = wrapped_func.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0]
            if cls not in component_graph_fn_registry:
                component_graph_fn_registry[cls] = list()
                component_graph_fn_registry[cls].append(graph_fn_rec)

        return _graph_fn_wrapper

    if graph_fn is None:
        return decorator_func
    else:
        return decorator_func(graph_fn)


def define_api_method(component, api_method_record, copy_record=True):
    """
    Registers an API-method with a Component instance.

    Args:
        component (Component): The Component object to register the API method with.
        api_method_record (APIMethodRecord): The APIMethodRecord describing the to-be-registered API-method.
        copy_record (bool): Whether to deepcopy the APIMethodRecord prior to handing it to the Component for storing.
    """
    # Deep copy the record (in case this got registered the normal way with via decorating a class method).
    if copy_record:
        api_method_record = copy.deepcopy(api_method_record)
    api_method_record.component = component

    # Raise errors if `name` already taken in this Component.
    if not api_method_record.ok_to_overwrite:
        # There already is an API-method with that name.
        if api_method_record.name in component.api_methods:
            raise RLGraphError("API-method with name '{}' already defined!".format(api_method_record.name))
        # There already is another object property with that name (avoid accidental overriding).
        elif not api_method_record.is_class_method and getattr(component, api_method_record.name, None) is not None:
            raise RLGraphError(
                "Component '{}' already has a property called '{}'. Cannot define an API-method with "
                "the same name!".format(component.name, api_method_record.name)
            )

    # Do not build this API as per ctor instructions.
    if api_method_record.name in component.switched_off_apis:
        return

    component.synthetic_methods.add(api_method_record.name)
    setattr(component, api_method_record.name, api_method_record.wrapper_func.__get__(component, component.__class__))
    setattr(api_method_record.wrapper_func, "__name__", api_method_record.name)

    component.api_methods[api_method_record.name] = api_method_record

    # Direct callable for eager/define by run.
    component.api_fn_by_name[api_method_record.name] = api_method_record.wrapper_func

    # Update the api_method_inputs dict (with empty Spaces if not defined yet).
    skip_args = 1  # self
    skip_args += (api_method_record.is_graph_fn_wrapper and api_method_record.add_auto_key_as_first_param)
    param_list = list(inspect.signature(api_method_record.func).parameters.values())[skip_args:]

    for param in param_list:
        component.api_methods[api_method_record.name].input_names.append(param.name)
        if param.name not in component.api_method_inputs:
            # This param has a default value.
            if param.default != inspect.Parameter.empty:
                # Default is None. Set to "flex" (to signal that this Space is not needed for input-completeness)
                # and wait for first call using this parameter (only then set it to that Space).
                if param.default is None:
                    component.api_method_inputs[param.name] = "flex"
                # Default is some python value (e.g. a bool). Use that are the assigned Space.
                else:
                    space = get_space_from_op(param.default)
                    component.api_method_inputs[param.name] = space
            # This param is an *args param. Store as "*flex". Then with upcoming API calls, we determine the Spaces
            # for the single items in *args and set them under "param[0]", "param[1]", etc..
            elif param.kind == inspect.Parameter.VAR_POSITIONAL:
                component.api_method_inputs[param.name] = "*flex"
            # This param is a **kwargs param. Store as "**flex". Then with upcoming API calls, we determine the Spaces
            # for the single items in **kwargs and set them under "param[some-key]", "param[some-other-key]", etc..
            elif param.kind == inspect.Parameter.VAR_KEYWORD:
                component.api_method_inputs[param.name] = "**flex"
            # Normal POSITIONAL_ONLY parameter. Store as None (needed) for now.
            else:
                component.api_method_inputs[param.name] = None


def define_graph_fn(component, graph_fn_record, copy_record=True):
    """
    Registers a graph_fn with a Component instance.

    Args:
        component (Component): The Component object to register the graph function with.
        graph_fn_record (GraphFnRecord): The GraphFnRecord describing the to-be-registered graph function.
        copy_record (bool): Whether to deepcopy the GraphFnRecord prior to handing it to the Component for storing.
    """
    # Deep copy the record (in case this got registered the normal way with via decorating a class method).
    if copy_record is True:
        graph_fn_record = copy.deepcopy(graph_fn_record)

    graph_fn_record.component = component

    # Raise errors if `name` already taken in this Component.
    # There already is a graph_fn with that name.
    if graph_fn_record.name in component.graph_fns:
        raise RLGraphError("Graph-Fn with name '{}' already defined!".format(graph_fn_record.name))
    # There already is another object property with that name (avoid accidental overriding).
    elif not graph_fn_record.is_class_method and getattr(component, graph_fn_record.name, None) is not None:
        raise RLGraphError(
            "Component '{}' already has a property called '{}'. Cannot define a Graph-Fn with "
            "the same name!".format(component.name, graph_fn_record.name)
        )

    setattr(component, graph_fn_record.name, graph_fn_record.wrapper_func.__get__(component, component.__class__))
    setattr(graph_fn_record.func, "__self__", component)

    component.graph_fns[graph_fn_record.name] = graph_fn_record


def graph_fn_wrapper(component, wrapped_func, returns, options, *args, **kwargs):
    """
    Executes a dry run through a graph_fn (without calling it) just generating the empty
    op-record-columns around the graph_fn (incoming and outgoing). Except if the GraphBuilder
    is already in the "building" phase, in which case the graph_fn is actually called.

    Args:
        component (Component): The Component that this graph_fn belongs to.
        wrapped_func (callable): The graph_fn to be called during the build process.
        returns (Optional[int]): The number of return values of the graph_fn.

        options (Dict): Dict with the following keys (optionally) set:
            - flatten_ops (Union[bool,Set[str]]): Whether to flatten all or some DataOps by creating
            a FlattenedDataOp (with automatic key names).
            Can also be a set of in-Socket names to flatten explicitly (True for all).
            Default: True.
            - split_ops (Union[bool,Set[str]]): Whether to split all or some of the already flattened DataOps
            and send the SingleDataOps one by one through the graph_fn.
            Example: Spaces=A=Dict (container), B=int (primitive)
            The graph_fn should then expect for each primitive Space in A:
            _graph_fn(primitive-in-A (Space), B (int))
            NOTE that B will be the same in all calls for all primitive-in-A's.
            Default: True.
            - add_auto_key_as_first_param (bool): If `split_ops` is not False, whether to send the
            automatically generated flat key as the very first parameter into each call of the graph_fn.
            Example: Spaces=A=float (primitive), B=Tuple (container)
            The graph_fn should then expect for each primitive Space in B:
            _graph_fn(key, A (float), primitive-in-B (Space))
            NOTE that A will be the same in all calls for all primitive-in-B's.
            The key can now be used to index into variables equally structured as B.
            Has no effect if `split_ops` is False.
            Default: False.

        \*args (Union[DataOpRecord,np.array,numeric]): The DataOpRecords to be used for calling the method.
    """
    flatten_ops = options.pop("flatten_ops", False)
    split_ops = options.pop("split_ops", False)
    add_auto_key_as_first_param = options.pop("add_auto_key_as_first_param", False)
    requires_variable_completeness = options.pop("requires_variable_completeness", False)

    # Store a graph_fn record in this component for better in/out-op-record-column reference.
    if wrapped_func.__name__ not in component.graph_fns:
        component.graph_fns[wrapped_func.__name__] = GraphFnRecord(
            func=wrapped_func, wrapper_func=graph_fn_wrapper, component=component,
            requires_variable_completeness=requires_variable_completeness
        )

    # Generate in-going op-rec-column.
    in_graph_fn_column = DataOpRecordColumnIntoGraphFn(
        component=component, graph_fn=wrapped_func,
        flatten_ops=flatten_ops, split_ops=split_ops,
        add_auto_key_as_first_param=add_auto_key_as_first_param,
        requires_variable_completeness=requires_variable_completeness,
        args=args, kwargs=kwargs
    )
    # Add the column to the `graph_fns` record.
    component.graph_fns[wrapped_func.__name__].in_op_columns.append(in_graph_fn_column)

    # We are already building: Actually call the graph_fn after asserting that its Component is input-complete.
    if component.graph_builder and component.graph_builder.phase == "building":
        # Assert input-completeness of Component (if not already, then after this graph_fn/Space update).
        # if self.input_complete is False:
        # Check Spaces and create variables.
        component.graph_builder.build_component_when_input_complete(component)
        if component.input_complete is False:
            raise RLGraphInputIncompleteError(component)
        # If we are calling a variables-requiring graph_fn -> make sure we are also variable-complete.
        if requires_variable_completeness is True and component.variable_complete is False:
            raise RLGraphVariableIncompleteError(component)
        # Call the graph_fn (only if not already done so by build above (e.g. `variables()`).
        if in_graph_fn_column.out_graph_fn_column is None:
            assert in_graph_fn_column.already_sent is False
            out_graph_fn_column = component.graph_builder.run_through_graph_fn_with_device_and_scope(in_graph_fn_column)
        else:
            out_graph_fn_column = in_graph_fn_column.out_graph_fn_column
        # Check again, in case we are now also variable-complete.
        component.graph_builder.build_component_when_input_complete(component)

    # We are still in the assembly phase: Don't actually call the graph_fn. Only generate op-rec-columns
    # around it (in-coming and out-going).
    else:
        # Create 2 op-record columns, one going into the graph_fn and one getting out of there and link
        # them together via the graph_fn (w/o calling it).
        # TODO: remove when we have numpy-based Components (then we can do test calls to infer everything automatically)
        if wrapped_func.__name__ in component.graph_fn_num_outputs:
            num_graph_fn_return_values = component.graph_fn_num_outputs[wrapped_func.__name__]
        elif returns is not None:
            num_graph_fn_return_values = returns
        else:
            num_graph_fn_return_values = util.get_num_return_values(wrapped_func)
        component.logger.debug("Graph_fn has {} return values (inferred).".format(
            wrapped_func.__name__, num_graph_fn_return_values)
        )
        # If in-column is empty, add it to the "empty in-column" set.
        if len(in_graph_fn_column.op_records) == 0:
            component.no_input_graph_fn_columns.add(in_graph_fn_column)

        # Generate the out-op-column from the number of return values (guessed during assembly phase or
        # actually measured during build phase).
        out_graph_fn_column = DataOpRecordColumnFromGraphFn(
            num_op_records=num_graph_fn_return_values,
            component=component, graph_fn_name=wrapped_func.__name__,
            in_graph_fn_column=in_graph_fn_column
        )

        in_graph_fn_column.out_graph_fn_column = out_graph_fn_column

    component.graph_fns[wrapped_func.__name__].out_op_columns.append(out_graph_fn_column)

    return_ops = False
    for stack_item in inspect.stack()[1:]:  # skip current frame
        # If we hit an API-method call -> return op-recs.
        if stack_item[3] == "api_method_wrapper" and re.search(r'decorators\.py$', stack_item[1]):
            break
        # If we hit a graph_fn call -> return ops.
        elif stack_item[3] == "run_through_graph_fn" and re.search(r'graph_builder\.py$', stack_item[1]):
            return_ops = True
            break

    if return_ops is True:  #re.match(r'^_graph_fn_.+|<lambda>$', stack[2][3]) and out_graph_fn_column.op_records[0].op is not None:
        assert out_graph_fn_column.op_records[0].op is not None,\
            "ERROR: Cannot return ops (instead of op-recs) if ops are still None!"
        if len(out_graph_fn_column.op_records) == 1:
            return out_graph_fn_column.op_records[0].op
        else:
            return tuple([op_rec.op for op_rec in out_graph_fn_column.op_records])
    else:
        if len(out_graph_fn_column.op_records) == 1:
            return out_graph_fn_column.op_records[0]
        else:
            return tuple(out_graph_fn_column.op_records)


def _sanity_check_call_parameters(self, params, method, method_type, add_auto_key_as_first_param):
    raw_signature_parameters = inspect.signature(method).parameters
    actual_params = list(raw_signature_parameters.values())
    if add_auto_key_as_first_param is True:
        actual_params = actual_params[1:]
    if len(params) != len(actual_params):
        # Check whether the last arg is var_positional (e.g. *inputs; in that case it's ok if the number of params
        # is larger than that of the actual graph_fn params or its one smaller).
        if actual_params[-1].kind == inspect.Parameter.VAR_POSITIONAL and (len(params) > len(actual_params) > 0 or
                                                                           len(params) == len(actual_params) - 1):
            pass
        # Some actual params have default values: Number of given params must be at least as large as the number
        # of non-default actual params but maximally as large as the number of actual_parameters.
        elif len(actual_params) >= len(params) >= sum(
                [p.default is inspect.Parameter.empty for p in actual_params]):
            pass
        else:
            raise RLGraphError(
                "ERROR: {} '{}/{}' has {} input-parameters, but {} ({}) were being provided in the "
                "`Component.call` method!".format(method_type, self.name, method.__name__,
                                                  len(actual_params), len(params), params)
            )


def _sanity_check_decorator_options(flatten_ops, split_ops, add_auto_key_as_first_param):
    if split_ops:
        assert flatten_ops,\
            "ERROR in decorator options: `split_ops` cannot be True if `flatten_ops` is False!"

    if add_auto_key_as_first_param:
        assert split_ops,\
            "ERROR in decorator options: `add_auto_key_as_first_param` cannot be True if `split_ops` is False!"
