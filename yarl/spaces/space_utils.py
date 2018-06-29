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

import re
import numpy as np

from yarl.utils.util import YARLError, dtype, get_shape, force_list
from yarl.utils.ops import SingleDataOp, ContainerDataOp, DataOpDict, DataOpTuple, FlattenedDataOp, DataOpRecord
from yarl.spaces import *


def get_space_from_op(op):
    """
    Tries to re-create a Space object given some DataOp.
    This is useful for shape inference when passing a Socket's ops through a GraphFunction and
    auto-inferring the resulting shape/Space.

    Args:
        op (DataOp): The op to create a corresponding Space for.

    Returns:
        Space: The inferred Space object.
    """
    # a Dict
    if isinstance(op, dict):  # DataOpDict
        spec = dict()
        add_batch_rank = False
        for k, v in op.items():
            spec[k] = get_space_from_op(v)
            if spec[k].has_batch_rank:
                add_batch_rank = True
        return Dict(spec, add_batch_rank=add_batch_rank)
    # a Tuple
    elif isinstance(op, tuple):  # DataOpTuple
        spec = list()
        add_batch_rank = False
        for i in op:
            spec.append(get_space_from_op(i))
            if spec[-1].has_batch_rank:
                add_batch_rank = True
        return Tuple(spec, add_batch_rank=add_batch_rank)
    # primitive Space -> infer from op dtype and shape
    else:
        # Simple constant value DataOp (python type or an np.ndarray).
        if isinstance(op, SingleDataOp) and op.constant_value is not None:
            value = op.constant_value
            if isinstance(value, np.ndarray):
                return BoxSpace.from_spec(spec=dtype(str(value.dtype), "np"), shape=value.shape)
        # No Space: e.g. the tf.no_op, a distribution (anything that's not a tensor).
        elif hasattr(op, "dtype") is False or not hasattr(op, "get_shape"):
            return 0
        # Some tensor: can be converted into a BoxSpace.
        else:
            shape = get_shape(op)
            # Unknown shape (e.g. a cond op).
            if shape is None:
                return 0
            add_batch_rank = False
            if shape is not () and shape[0] is None:
                shape = shape[1:]
                add_batch_rank = True

            # a FloatBox
            if op.dtype.base_dtype == dtype("float"):
                return FloatBox(shape=shape, add_batch_rank=add_batch_rank)
            # an IntBox
            elif op.dtype.base_dtype == dtype("int"):
                return IntBox(shape=shape, add_batch_rank=add_batch_rank)  # low, high=dummy values
            # a BoolBox
            elif op.dtype.base_dtype == dtype("bool"):
                return BoolBox(add_batch_rank=add_batch_rank)

    raise YARLError("ERROR: Cannot derive Space from op '{}' (unknown type?)!".format(op))


def flatten_op(op, scope_="", list_=None):
    """
    Flattens a single ContainerDataOp or a native python dict/tuple into a FlattenedDataOp with auto-key generation.

    Args:
        op (Union[ContainerDataOp,dict,tuple]): The item to flatten.
        scope_ (str): The recursive scope for auto-key generation.
        list_ (list): The list of tuples (key, value) to be converted into the final FlattenedDataOp.

    Returns:
        FlattenedDataOp: The flattened representation of the op.
    """
    ret = False

    # Are we in the non-recursive (first) call?
    if list_ is None:
        # Flatten a SingleDataOp -> return FlattenedDataOp with only-key=""
        if not isinstance(op, (ContainerDataOp, dict, tuple)):
            return FlattenedDataOp([("", op)])
        list_ = list()
        ret = True

    if isinstance(op, dict):
        scope_ += "/"
        for key in sorted(op.keys()):
            flatten_op(op[key], scope_=scope_ + key, list_=list_)
    elif isinstance(op, tuple):
        scope_ += "/" + FLAT_TUPLE_OPEN
        for i, c in enumerate(op):
            flatten_op(c, scope_=scope_ + str(i) + FLAT_TUPLE_CLOSE, list_=list_)
    else:
        assert not isinstance(op, (dict, tuple, list))
        list_.append((scope_, op))

    # Non recursive (first) call -> Return the final FlattenedDataOp.
    if ret:
        return FlattenedDataOp(list_)


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

    for op_name, op_val in op.items():
        parent_structure = None
        parent_key = None
        current_structure = None
        type_ = None

        op_key_list = op_name[1:].split("/")  # skip 1st char (/)
        for sub_key in op_key_list:
            mo = re.match(r'^{}(\d+){}$'.format(FLAT_TUPLE_OPEN, FLAT_TUPLE_CLOSE), sub_key)
            if mo:
                type_ = list
                idx = int(mo.group(1))
            else:
                type_ = DataOpDict
                idx = sub_key

            if current_structure is None:
                if base_structure is None:
                    base_structure = [None] if type_ == list else DataOpDict()
                current_structure = base_structure
            elif parent_key is not None:
                if isinstance(parent_structure, list) and parent_structure[parent_key] is None or \
                        isinstance(parent_structure, DataOpDict) and parent_key not in parent_structure:
                    current_structure = [None] if type_ == list else DataOpDict()
                    parent_structure[parent_key] = current_structure
                else:
                    current_structure = parent_structure[parent_key]
                    if type_ == list and len(current_structure) == idx:
                        current_structure.append(None)

            parent_structure = current_structure
            parent_key = idx
            if isinstance(parent_structure, list) and len(parent_structure) == parent_key:
                parent_structure.append(None)

        if type_ == list and len(current_structure) == parent_key:
            current_structure.append(None)
        current_structure[parent_key] = op_val

    # Deep conversion from list to tuple.
    return deep_tuple(base_structure)


def convert_op_to_op_record(op, group=None):
    """
    Translates all elements in `ops` into DataOpRecords using the given labels and returns the tuple of
    DataOpRecords.

    Args:
        op (op): A primitive op (e.g. tf op) to convert into a DataOpRecord.
        group (Optional[int]): An op group ID to give to the generated op record.

    Returns:
        DataOpRecord: The given op wrapped as DataOpRecord (with the given group ID).
    """
    # Convert to SingleDataOp with constant value if op is some primitive.
    if isinstance(op, (int, float, bool, np.ndarray)):
        op = SingleDataOp(constant_value=op)
    return DataOpRecord(op, group=group)


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


def sanity_check_space(
        space, allowed_types=None, non_allowed_types=None, must_have_batch_rank=None, must_have_categories=None,
        rank=None, num_categories=None
):
    """
    Sanity checks a given Space for certain criteria and raises exceptions if they are not met.

    Args:
        space (Space): The Space object to check.
        allowed_types (Optional[List[type]]): A list of types that this Space must be an instance of.
        non_allowed_types (Optional[List[type]]): A list of type that this Space must not be an instance of.
        must_have_batch_rank (Optional[bool]): Whether the Space  must (True) or must not (False) have the
            `has_batch_rank` property set to True. None, if it doesn't matter.
        must_have_categories (Optional[bool]): For IntBoxes, whether the Space must (True) or must not (False) have
            global bounds with `num_categories` > 0. None, if it doesn't matter.
        rank (Optional[int,tuple]): An int or a tuple (min,max) range within which the Space's rank must lie.
            None if it doesn't matter.
        num_categories (Optional[int,tuple]): An int or a tuple (min,max) range within which the Space's
            `num_categories` rank must lie. Only valid for IntBoxes.
            None if it doesn't matter.

    Raises:
        YARLError: Various YARLErrors, if any of the conditions is not met.
    """
    # Check the types.
    if allowed_types is not None:
        if not isinstance(space, tuple(allowed_types)):
            raise YARLError("ERROR: Space ({}) is not an instance of {}!".format(space, allowed_types))

    if non_allowed_types is not None:
        if isinstance(space, tuple(non_allowed_types)):
            raise YARLError("ERROR: Space ({}) must not be an instance of {}!".format(space, non_allowed_types))

    if must_have_batch_rank is not None:
        if space.has_batch_rank != must_have_batch_rank:
            # Last chance: Check for rank >= 2, that would be ok as well.
            if must_have_batch_rank is True and len(space.get_shape(with_batch_rank=True)) >= 2:
                pass
            # Something is wrong.
            elif space.has_batch_rank is True:
                raise YARLError("ERROR: Space ({}) has a batch rank, but is not allowed to!".format(space))
            else:
                raise YARLError("ERROR: Space ({}) does not have a batch rank, but must have one!".format(space))

    if must_have_categories is not None:
        if not isinstance(space, IntBox):
            raise YARLError("ERROR: Space ({}) is not an IntBox. Only IntBox Spaces can have categories!".format(space))
        elif space.global_bounds is False:
            raise YARLError("ERROR: Space ({}) must have categories (globally valid value bounds)!".format(space))

    if rank is not None:
        if isinstance(rank, int):
            if space.rank != rank:
                raise YARLError("ERROR: Space ({}) has rank {}, but must have rank {}!".format(space, space.rank, rank))
        elif not ((rank[0] or 0) <= space.rank <= (rank[1] or float("inf"))):
            raise YARLError("ERROR: Space ({}) has rank {}, but its rank must be between {} and "
                            "{}!".format(space, space.rank, rank[0], rank[1]))

    if num_categories is not None:
        if not isinstance(space, IntBox):
            raise YARLError("ERROR: Space ({}) is not an IntBox. Only IntBox Spaces can have categories!".format(space))
        elif space.num_categories is None or space.num_categories == 0:
            if isinstance(num_categories, int):
                if space.num_categories != num_categories:
                    raise YARLError("ERROR: Space ({}) has `num_categories` {}, but must have {}!".
                                    format(space, space.num_categories, num_categories))
            elif not ((num_categories[0] or 0) <= space.num_categories <= (num_categories[1] or float("inf"))):
                raise YARLError("ERROR: Space ({}) has `num_categories` {}, but this value must be between {} and "
                                "{}!".format(space, space.num_categories, num_categories[0], num_categories[1]))
