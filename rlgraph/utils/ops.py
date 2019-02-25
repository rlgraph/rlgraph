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

from collections import OrderedDict
import re


# Defines how to generate auto-keys for flattened Tuple-Space items.
# _T\d+_
FLAT_TUPLE_OPEN = "_T"
FLAT_TUPLE_CLOSE = "_"


class TraceContext(object):
    """
    Contains static trace context. Used to reconstruct data-flow in cases where normal
    stack-frame inspection fails, e.g. because when calling from within a lambda. To be expanded
    with more runtime
    """

    # Prior caller.
    PREV_CALLER = None

    # Define by run build tracing.
    DEFINE_BY_RUN_CONTEXT = None


class DataOp(object):
    """
    The basic class for any Socket-held operation or variable, or collection thereof.
    Each Socket (in or out) holds either one DataOp or a set of alternative DataOps.
    """
    pass


class SingleDataOp(DataOp):
    """
    A placeholder class for a simple (non-container) Tensor going into a GraphFunction or coming out of a GraphFunction,
    or a tf.no_op-like item.
    """
    pass


class ContainerDataOp(DataOp):
    """
    A placeholder class for any DataOp that's not a SingleDataOp, but a (possibly nested) container structure
    containing SingleDataOps as leave nodes.
    """
    def flat_key_lookup(self, flat_key):
        """
        Returns an element within this DataOp following a given flat-key.

        Args:
            flat_key ():

        Returns:

        """
        key_sequence = flat_key.split("/")
        result = self
        for key in key_sequence:
            mo = re.match(r'^{}(\d+){}$'.format(FLAT_TUPLE_OPEN, FLAT_TUPLE_CLOSE), key)
            # Tuple
            if mo is not None:
                result = result[int(mo.group(1))]
            # Dict
            else:
                result = result[key]
        return result


class DataOpDict(ContainerDataOp, dict):
    """
    A hashable dict that's used to make (possibly nested) dicts of SingleDataOps hashable, so that
    we can store them in sets and use them as lookup keys in other dicts.
    Dict() Spaces produce DataOpDicts when methods like `get_variable` are called on them.
    """
    def __hash__(self):
        """
        Hash based on sequence of sorted items (keys are all strings, values are always other DataOps).
        """
        return hash(tuple(sorted(self.items())))


class DataOpTuple(ContainerDataOp, tuple):
    """
    A simple wrapper for a (possibly nested) tuple that contains other DataOps.
    """
    def __new__(cls, *components):
        if isinstance(components[0], (list, tuple)):
            assert len(components) == 1
            components = components[0]

        return tuple.__new__(cls, components)


class FlattenedDataOp(DataOp, OrderedDict):
    """
    An OrderedDict-type placeholder class that only contains str as keys and SingleDataOps
    (as opposed to ContainerDataOps) as values.
    """
    # TODO: enforce str as keys?
    pass


def flatten_op(op, key_scope="", op_tuple_list=None, scope_separator_at_start=False):
    """
    Flattens a single ContainerDataOp or a native python dict/tuple into a FlattenedDataOp with auto-key generation.

    Args:
        op (Union[ContainerDataOp,dict,tuple]): The item to flatten.
        key_scope (str): The recursive scope for auto-key generation.
        op_tuple_list (list): The list of tuples (key, value) to be converted into the final FlattenedDataOp.
        scope_separator_at_start (bool): If to prepend a scope separator before the first key in a
            recursive structure. Default false.

    Returns:
        FlattenedDataOp: The flattened representation of the op.
    """
    ret = False

    # Are we in the non-recursive (first) call?
    if op_tuple_list is None:
        # Flatten a SingleDataOp -> return FlattenedDataOp with only-key=""
        if not isinstance(op, (ContainerDataOp, dict, tuple)):
            return FlattenedDataOp([("", op)])
        op_tuple_list = []
        ret = True

    if isinstance(op, dict):
        if scope_separator_at_start:
            key_scope += "/"
        else:
            key_scope = ""
        for key in sorted(op.keys()):
            # Make sure we have no double slashes from flattening an already FlattenedDataOp.
            scope = (key_scope[:-1] if len(key) == 0 or key[0] == "/" else key_scope) + key
            flatten_op(op[key], key_scope=scope, op_tuple_list=op_tuple_list, scope_separator_at_start=True)
    elif isinstance(op, tuple):
        if scope_separator_at_start:
            key_scope += "/" + FLAT_TUPLE_OPEN
        else:
            key_scope += "" + FLAT_TUPLE_OPEN
        for i, c in enumerate(op):
            flatten_op(c, key_scope=key_scope + str(i) + FLAT_TUPLE_CLOSE, op_tuple_list=op_tuple_list,
                       scope_separator_at_start=True)
    else:
        assert not isinstance(op, (dict, tuple))
        op_tuple_list.append((key_scope, op))

    # Non recursive (first) call -> Return the final FlattenedDataOp.
    if ret:
        return FlattenedDataOp(op_tuple_list)


def unflatten_op(op):
    """
    Takes a FlattenedDataOp with auto-generated keys and returns the corresponding
    unflattened DataOp.
    If the only key in the input FlattenedDataOp is "", it returns the SingleDataOp under
    that key.

    Args:
        op (dict): The item to be unflattened (re-nested) into any DataOp. Usually a FlattenedDataOp, but can also
            be a plain dict.

    Returns:
        DataOp: The unflattened (re-nested) item.
    """
    # Special case: FlattenedDataOp with only 1 SingleDataOp (key="").
    if len(op) == 1 and "" in op:
        return op[""]

    # Normal case: FlattenedDataOp that came from a ContainerItem.
    base_structure = None

    op_names = sorted(op.keys())
    for op_name in op_names:
        op_val = op[op_name]
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
                op_type = DataOpDict
                idx = sub_key

            if current_structure is None:
                if base_structure is None:
                    base_structure = [None] if op_type == list else DataOpDict()
                current_structure = base_structure
            elif parent_key is not None:
                # DEBUG:
                #if isinstance(parent_structure, list) and len(parent_structure) <= parent_key:
                #    print("WARNING: parent_structure={} parent_key={} to-be-flattened-op={}".format(parent_structure, parent_key, op))
                # END: DEBUG

                if (isinstance(parent_structure, list) and (parent_structure[parent_key] is None)) or \
                        (isinstance(parent_structure, DataOpDict) and parent_key not in parent_structure):
                    current_structure = [None] if op_type == list else DataOpDict()
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
    return deep_tuple(base_structure)


def deep_tuple(x):
    """
    Converts an input list of list (of list, etc..) into the respective nested DataOpTuple.

    Args:
        x (list): The input list to be converted into a tuple.

    Returns:
        tuple: The corresponding tuple to x.
    """
    # A list -> convert to DataOpTuple.
    if isinstance(x, list):
        return DataOpTuple(list(map(deep_tuple, x)))
    # A dict -> leave type as is and keep converting recursively.
    elif isinstance(x, dict):
        # type(x) b/c x could be DataOpDict as well.
        return type(x)(dict(map(lambda i: (i[0], deep_tuple(i[1])), x.items())))
    # A primitive -> keep as is.
    else:
        return x


def is_constant(op):
    return type(op).__name__ in ["int", "float", "bool", "ndarray"]
