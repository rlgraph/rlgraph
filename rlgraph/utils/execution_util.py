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

import re
from collections import OrderedDict

from rlgraph import get_backend
from rlgraph.utils.ops import FLAT_TUPLE_OPEN, FLAT_TUPLE_CLOSE, deep_tuple, FlattenedDataOp

if get_backend() == "pytorch":
    import torch


def print_call_chain(profile_data, sort=True, filter_threshold=None):
    """
    Prints a component call chain stdout. Useful to analyze define by run performance.

    Args:
        profile_data (list): Component file data.
        sort (bool): If true, sorts call sorted by call duration.
        filter_threshold (Optional[float]): Optionally specify an execution threshold in seconds (e.g. 0.01).
            All call entries below the threshold be dropped from the printout.
    """
    original_length = len(profile_data)
    if filter_threshold is not None:
        assert isinstance(filter_threshold, float), "ERROR: Filter threshold must be float but is {}.".format(
            type(filter_threshold))
        profile_data = [data for data in profile_data if data[2] > filter_threshold]
    if sort:
        res = sorted(profile_data, key=lambda v: v[2], reverse=True)
        print("Call chain sorted by runtime ({} calls, {} before filter):".
              format(len(profile_data), original_length))
        for v in res:
            print("{}.{}: {} s".format(v[0], v[1], v[2]))
    else:
        print("Directed call chain ({} calls, {} before filter):".format(len(profile_data), original_length))
        for i in range(len(profile_data) - 1):
            v = profile_data[i]
            print("({}.{}: {} s) ->".format(v[0], v[1], v[2]))
        v = profile_data[-1]
        print("({}.{}: {} s)".format(v[0], v[1], v[2]))


def define_by_run_flatten(container, key_scope="", tensor_tuple_list=None, scope_separator_at_start=False):
    """
    Flattens a native python dict/tuple into a flat dict with auto-key generation. Run-time equivalent
    to build-time flatten operation.

    Args:
        container (Union[dict,tuple]): Container  to flatten.
        key_scope (str): The recursive scope for auto-key generation.
        tensor_tuple_list (list): The list of tuples (key, value) to be converted into the final results.
        scope_separator_at_start (bool): If to prepend a scope separator before the first key in a
            recursive structure. Default false.

    Returns:
        Dict: Flattened container.
    """
    ret = False

    # Are we in the non-recursive (first) call?
    if tensor_tuple_list is None:
        tensor_tuple_list = []
        if not isinstance(container, (dict, tuple)):
            return OrderedDict([("", container)])
        ret = True

    if isinstance(container, dict):
        if scope_separator_at_start:
            key_scope += "/"
        else:
            key_scope = ""
        for key in sorted(container.keys()):
            # Make sure we have no double slashes from flattening an already FlattenedDataOp.
            scope = (key_scope[:-1] if len(key) == 0 or key[0] == "/" else key_scope) + key
            define_by_run_flatten(container[key], key_scope=scope, tensor_tuple_list=tensor_tuple_list, scope_separator_at_start=True)
    elif isinstance(container, tuple):
        if scope_separator_at_start:
            key_scope += "/" + FLAT_TUPLE_OPEN
        else:
            key_scope += "" + FLAT_TUPLE_OPEN
        for i, c in enumerate(container):
            define_by_run_flatten(c, key_scope=key_scope + str(i) + FLAT_TUPLE_CLOSE, tensor_tuple_list=tensor_tuple_list,
                                  scope_separator_at_start=True)
    else:
        assert not isinstance(container, (dict, tuple))
        tensor_tuple_list.append((key_scope, container))

    # Non recursive (first) call -> Return the final dict.
    if ret:
        return OrderedDict(tensor_tuple_list)


def define_by_run_split_args(add_auto_key_as_first_param, *args, **kwargs):
    """
    Splits any container in *args and **kwargs and collects them to be evaluated
    one by one by a graph_fn. If more than one container inputs exists in *args and **kwargs,
    these must have the exact same keys.

    Note that this does not perform any checks on aligned keys, these were performed at build time.

    Args:
        add_auto_key_as_first_param (bool): Add auto-key as very first parameter in each
        returned parameter tuple.
        *args (op): args to split.
        **kwargs (op): kwargs to split.

    Returns:
        Union[OrderedDict,Tuple]]: The sorted parameter tuples (by flat-key) to use as inputs.
    """

    # Collect Dicts for checking their keys (must match).

    flattened_args = []
    # Use the loop below to unwrap single-key dicts with default keys to be used if no
    # true splitting is happening.
    unwrapped_args = [""] if add_auto_key_as_first_param is True else []
    lead_container_arg = None
    if get_backend() == "pytorch":
        for arg in args:
            # Convert raw torch tensors: during flattening, we do not flatten single tensors
            # to avoid flattening altogether if there are strictly raw tensors.
            if isinstance(arg, torch.Tensor):
                flattened_args.append({"": arg})
                unwrapped_args.append(arg)
                # Append raw tensor.
            elif isinstance(arg, dict):
                if len(arg) > 1 or "" not in arg:
                    flattened_args.append(arg)
                    # Use first encountered container arg.
                    if lead_container_arg is None:
                        lead_container_arg = arg
                else:
                    unwrapped_args.append(arg[""])

    # One or more dicts: Split the calls.
    if len(flattened_args) > 0 and lead_container_arg is not None:
        # Re-create our iterators.
        collected_call_params = OrderedDict()

        for key in lead_container_arg.keys():
            # Prep input params for a single call.
            params = [key] if add_auto_key_as_first_param is True else []

            for arg in flattened_args:
                params.append(arg[key] if key in arg else arg[""])

            # Add kwarg_ops
            for kwarg_key, kwarg_op in kwargs.items():
                params.append(tuple([
                    kwarg_key,
                    kwargs[kwarg_key][key] if key in kwargs[kwarg_key] else kwargs[kwarg_key][""]
                ]))

            collected_call_params[key] = params[0] if len(params) == 1 else params
        return collected_call_params
    # We don't have any containers: No splitting possible. Return args and kwargs as is.
    else:
        return unwrapped_args, {key: value[""] for key, value in kwargs.items()}


def define_by_run_unflatten(result_dict):
    """
    Takes a dict with auto-generated keys and returns the corresponding
    unflattened dict.
    If the only key in the input dict is "", it returns the value under
    that key.

    Args:
        result_dict (dict): The item to be unflattened (re-nested).

    Returns:
        Dict: The unflattened (re-nested) item.
    """
    # Special case: Dict with only 1 value (key="")
    if len(result_dict) == 1 and "" in result_dict:
        return result_dict[""]

    # Normal case: OrderedDict that came from a ContainerItem.
    base_structure = None

    op_names = sorted(result_dict.keys())
    for op_name in op_names:
        op_val = result_dict[op_name]
        parent_structure = None
        parent_key = None
        current_structure = None
        op_type = None

        # N.b. removed this because we do not prepend / any more before first key.
        op_key_list = op_name.split("/")  # skip 1st char (/)
        for sub_key in op_key_list:
            mo = re.match(r'^{}(\d+){}$'.format(FLAT_TUPLE_OPEN, FLAT_TUPLE_CLOSE), sub_key)
            if mo:
                op_type = list
                idx = int(mo.group(1))
            else:
                op_type = OrderedDict
                idx = sub_key

            if current_structure is None:
                if base_structure is None:
                    base_structure = [None] if op_type == list else OrderedDict()
                current_structure = base_structure
            elif parent_key is not None:
                if (isinstance(parent_structure, list) and (parent_structure[parent_key] is None)) or \
                        (isinstance(parent_structure, OrderedDict) and parent_key not in parent_structure):
                    current_structure = [None] if op_type == list else OrderedDict()
                    parent_structure[parent_key] = current_structure
                else:
                    current_structure = parent_structure[parent_key]
                    if op_type == list and len(current_structure) == idx:
                        current_structure.append(None)

            parent_structure = current_structure
            parent_key = idx
            if isinstance(parent_structure, list) and len(parent_structure) == parent_key:
                parent_structure.append(None)

        if op_type == list and len(current_structure) == parent_key:
            current_structure.append(None)
        current_structure[parent_key] = op_val

    # Deep conversion from list to tuple.
    # TODO necessary in define by run?
    return deep_tuple(base_structure)


def define_by_run_unpack(args):
    """
    Unpacks potential nested FlattenedDataOp wrapper args.
    Args:
        args (any):

    Returns:
        any: Unpacked args.
    """
    if isinstance(args, FlattenedDataOp) and len(args) == 1:
        return next(iter(args.values()))
    elif isinstance(args, (list, tuple)):
        ret = []
        # Check list elements one by one.
        for value in args:
            if isinstance(value, FlattenedDataOp) and len(value) == 1:
                ret.append(next(iter(value.values())))
            else:
                ret.append(value)
        return ret
    else:
        return args
