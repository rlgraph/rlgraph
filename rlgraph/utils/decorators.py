# Copyright 2018 The Rlgraph Authors, All Rights Reserved.
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
import numpy as np
import re
import time

from rlgraph.spaces.space_utils import get_space_from_op
from rlgraph.utils.ops import APIMethodRecord, DataOpRecord, DataOpRecordColumnIntoAPIMethod, \
    DataOpRecordColumnFromAPIMethod
from rlgraph.utils.rlgraph_error import RLGraphError
from rlgraph.utils import util


component_api_registry = dict()


def api(api_method=None, *,
        component=None,
        flatten_ops=False, split_ops=False, add_auto_key_as_first_param=False, ok_to_overwrite=False,
        must_be_complete=True):
    """
    API-method decorator used to tag any Component's methods as API-methods.

    Args:
        api_method (callable): The actual function/method to tag as an API method.
        component (Component): The Component that the method should belong to.
        flatten_ops (bool): Whether to flatten input (Container) ops when calling this API.
        split_ops (bool):
        add_auto_key_as_first_param ():
        ok_to_overwrite ():
        must_be_complete ():

    Returns:

    """
    def decorator_func(api_method):

        def api_method_wrapper(self, *args, **kwargs):
            # Owner of method:
            method_owner = api_method.__self__  # type: Component
            # Check, whether the method-owner is either this Component or has this Component as parent.
            assert method_owner is self or self in method_owner.get_parents(), \
                "ERROR: Can only call API-method ({}/{}) on self ({}) or any sub-Components of self! Most likely, " \
                "{} has not been added to self.". \
                format(method_owner.global_scope, api_method.__name__, self.global_scope, method_owner.global_scope)
            # Try handing the graph-builder link to the owner Component (in case it's not set yet).
            if method_owner.graph_builder is None:
                method_owner.graph_builder = self.graph_builder

            return_ops = kwargs.pop("return_ops", False)

            # Direct evaluation of function.
            if self.execution_mode == "define_by_run":
                type(method_owner).call_count += 1

                # Name might not match, e.g. _graph_fn_apply vs apply.
                #  Check with owner if extra args needed.
                start = time.perf_counter()
                if api_method.__name__ in method_owner.api_methods and \
                        method_owner.api_methods[api_method.__name__].add_auto_key_as_first_param:
                    # Add auto key.
                    return_dict = api_method("", *args, **kwargs)
                else:
                    return_dict = api_method(*args, **kwargs)

                # Store runtime for this method.
                type(method_owner).call_times.append(  # Component.call_times
                    (method_owner.name, api_method.__name__, time.perf_counter() - start)
                )

                return return_dict

            api_method_rec = method_owner.api_methods[api_method.__name__]

            # Create op-record column to call API method with. Ignore None input params. These should not be sent
            # to the API-method.
            in_op_column = DataOpRecordColumnIntoAPIMethod(
                component=self, api_method_rec=api_method_rec, args=args, kwargs=kwargs
            )
            # Add the column to the API-method record.
            api_method_rec.in_op_columns.append(in_op_column)

            # Link from incoming op_recs into the new column or populate new column with ops/Spaces (this happens
            # if this call was made from within a graph_fn such that ops and Spaces are already known).
            all_args = [(i, a) for i, a in enumerate(args)] + [(k, v) for k, v in sorted(kwargs.items())]
            flex = None
            for i, value in all_args:
                # Named arg/kwarg -> get input_name from that and peel op_rec.
                if isinstance(i, str):
                    param_name = i
                # Positional arg -> get input_name from input_names list.
                else:
                    param_name = api_method_rec.input_names[i if flex is None else flex]

                # Var-positional arg, attach the actual position to input_name string.
                if method_owner.api_method_inputs[param_name] == "*flex":
                    if flex is None:
                        flex = i
                    param_name += "[{}]".format(i - flex)

                # We are already in building phase (params may be coming from inside graph_fn).
                if self.graph_builder is not None and self.graph_builder.phase == "building":
                    # Params are op-records -> link AND pass on actual ops/Spaces.
                    if isinstance(value, DataOpRecord):
                        in_op_column.op_records[i].op = value.op
                        in_op_column.op_records[i].space = method_owner.api_method_inputs[param_name] = \
                            get_space_from_op(value.op)
                        # op_rec.next.add(in_op_column.op_records[i])
                        # in_op_column.op_records[i].previous = op_rec
                    # Params are actual ops: Pass them (and Space) on to in-column records.
                    else:
                        in_op_column.op_records[i].op = value
                        in_op_column.op_records[i].space = method_owner.api_method_inputs[param_name] = \
                            get_space_from_op(value)
                    # Check input-completeness of Component (but not strict as we are only calling API, not a graph_fn).
                    if method_owner.input_complete is False:
                        # Check Spaces and create variables.
                        method_owner.graph_builder.build_component_when_input_complete(method_owner)

                # A DataOpRecord from the meta-graph.
                elif isinstance(value, DataOpRecord):
                    # op_rec.next.add(in_op_column.op_records[i])
                    # in_op_column.op_records[i].previous = op_rec
                    if param_name not in method_owner.api_method_inputs:
                        method_owner.api_method_inputs[param_name] = None

                # Fixed value (instead of op-record): Store the fixed value directly in the op.
                else:
                    in_op_column.op_records[i].op = np.array(value)
                    in_op_column.op_records[i].space = get_space_from_op(value)
                    if param_name not in method_owner.api_method_inputs or method_owner.api_method_inputs[
                        param_name] is None:
                        method_owner.api_method_inputs[param_name] = in_op_column.op_records[i].space
                    self.constant_op_records.add(in_op_column.op_records[i])

            # Now actually call the API method with that the correct *args and **kwargs from the new column and
            # create a new out-column with num-records == num-return values.
            name = api_method.__name__
            self.logger.debug("Calling api method {} with owner {}:".format(name, method_owner))
            # args = [op_rec for op_rec in in_op_column.op_records if op_rec.kwarg is None]
            # kwargs = {op_rec.kwarg: op_rec for op_rec in in_op_column.op_records if op_rec.kwarg is not None}
            return_dict = api_method(*args, **kwargs)

            # Process the results (push into a column).
            out_op_column = DataOpRecordColumnFromAPIMethod(
                component=self,
                api_method_name=name,
                return_dict=return_dict
            )

            # Link the returned ops to that new out-column.
            for i, key in enumerate(sorted(return_dict.keys())):
                # If we already have actual op(s) and Space(s), push them already into the
                # DataOpRecordColumnFromAPIMethod's records.
                if self.graph_builder is not None and self.graph_builder.phase == "building":
                    out_op_column.op_records[i].op = return_dict[key].op
                    out_op_column.op_records[i].space = return_dict[key].space
                # op_rec.next.add(out_op_column.op_records[i])
                # out_op_column.op_records[i].previous = op_rec
            # And append the new out-column to the api-method-rec.
            api_method_rec.out_op_columns.append(out_op_column)

            # Then return the op-records from the new out-column.
            # if len(out_op_column.op_records) == 1:
            #    return out_op_column.op_records[0]
            # else:
            #    return tuple(out_op_column.op_records)

            # return return_dict

            # Do we need to return the raw ops or the op-recs?
            # Direct parent caller is a `_graph_fn_...`: Return raw ops.
            stack = inspect.stack()
            if return_ops is True or re.match(r'^_graph_fn_.+$', stack[1][3]):
                return {key: value.op for key, value in return_dict.items()}
            # Parent caller is non-graph_fn: Return op-recs.
            else:
                return return_dict

        func_type = util.get_method_type(api_method)
        api_method_rec = APIMethodRecord(
            api_method, must_be_complete=must_be_complete, ok_to_overwrite=ok_to_overwrite,
            is_graph_fn_wrapper=func_type == "graph_fn", is_class_method=(component is None),
            flatten_ops=flatten_ops, split_ops=split_ops, add_auto_key_as_first_param=add_auto_key_as_first_param
        )

        # Registers the given method with the Component (if not already done so).
        if component is not None:
            define_api_method(component, api_method_rec)
        # Registers the given function with the Component sub-class so we can define it for each
        # constructed instance of that sub-class.
        else:
            cls = api_method.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0]
            if cls not in component_api_registry:
                component_api_registry[cls] = list()
            component_api_registry[cls].append(api_method_rec)

        return api_method_wrapper

    if api_method is None:
        return decorator_func
    else:
        return decorator_func(api_method)


def define_api_method(component, api_method_record):  #name, func, must_be_complete=True, ok_to_overwrite=False, **kwargs):
    """
    Creates a very basic graph_fn based API method for this Component.
    Alternative way for defining an API method directly in the Component via def.

    Args:
        name (Union[str,callable]): The name of the API method to create or one of the Component's method to
            create the API-method from.

        func (Optional[callable]): The graph_fn to wrap or the custom function to set as API-method.

        must_be_complete (bool): Whether this API-method must have all its incoming Spaces defined in order
            for the Component to count as "input-complete". Some API-methods may be still input-incomplete without
            affecting the Component's build process.

        ok_to_overwrite (bool): Whether to raise an Error if an API-method with that name or another property
            with that name already exists. Default: False.

    Keyword Args:
        flatten_ops (bool,Set[int]): See `self.call` for details.
        split_ops (bool,Set[int]): See `self.call` for details.
        add_auto_key_as_first_param (bool): See `self.call` for details.
    """
    # Deep copy the record (in case this got registered the normal way with via decorating a class method).
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

    # Function is a graph_fn: Build a simple wrapper API-method around it and name it `name`.
    if api_method_record.is_graph_fn_wrapper:
        def api_method(self, *inputs_, **kwargs_):
            # Mix in user provided kwargs with the setting ones from the original `define_api_method` call.
            #util.default_dict(kwargs_, kwargs)
            func_ = getattr(self, api_method_record.name)

            return func_(*inputs_, **kwargs_)

    # Function is a (custom) API-method. Register it with this Component.
    else:
        api_method = api_method_record.method

    component.synthetic_methods.add(api_method_record.name)
    setattr(component, api_method_record.name, api_method.__get__(component, component.__class__))
    setattr(api_method, "__self__", component)
    setattr(api_method, "__name__", api_method_record.name)

    component.api_methods[api_method_record.name] = api_method_record

    # Direct callable for eager/define by run.
    component.api_fn_by_name[api_method_record.name] = api_method

    # Update the api_method_inputs dict (with empty Spaces if not defined yet).
    # Note: Skip first param of graph_func's input param list if add-auto-key option is True (1st param would be
    # the auto-key then). Also skip if api_method is an unbound function (then 1st param is usually `self`).
    if (api_method_record.is_graph_fn_wrapper and api_method_record.add_auto_key_as_first_param) or \
            (not api_method_record.is_graph_fn_wrapper and type(api_method).__name__ == "function") or \
            api_method_record.is_class_method is True:
        skip_1st_arg = 1
    else:
        skip_1st_arg = 0
    param_list = list(inspect.signature(api_method_record.method).parameters.values())[skip_1st_arg:]
    _complement_api_method_registries(component, api_method_record.name, param_list)


def _complement_api_method_registries(self, api_method_name, param_list):
    self.api_methods[api_method_name].input_names = list()
    for param in param_list:
        self.api_methods[api_method_name].input_names.append(param.name)
        if param.name not in self.api_method_inputs:
            # This param has a default value.
            if param.default != inspect.Parameter.empty:
                # Default is None. Set to "flex" (to signal that this Space is not needed for input-completeness)
                # and wait for first call using this parameter (only then set it to that Space).
                if param.default is None:
                    self.api_method_inputs[param.name] = "flex"
                # Default is some python value (e.g. a bool). Use that are the assigned Space.
                else:
                    space = get_space_from_op(param.default)
                    self.api_method_inputs[param.name] = space
            # This param is an *args param. Store as "*flex". Then with upcoming API calls, we determine the Spaces
            # for the single items in *args and set them under "param[0]", "param[1]", etc..
            elif param.kind == inspect.Parameter.VAR_POSITIONAL:
                self.api_method_inputs[param.name] = "*flex"
            # Normal POSITIONAL_ONLY parameter. Store as None (needed) for now.
            else:
                self.api_method_inputs[param.name] = None


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
